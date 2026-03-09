# libero/lifelong/algos/skill_library_builder.py
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.metric import evaluate_one_task_success
from libero.lifelong.models.bc_action_expert import ActionExpert
from libero.lifelong.utils import safe_device, torch_save_model
from libero.lifelong.models.policy_head import DiffusionDist

class SkillLibraryBuilder(Sequential):
    def __init__(self, n_tasks, cfg):
        super().__init__(n_tasks, cfg)
        self.cfg = cfg

    def observe(self, data, expert=None):
        """计算 base + dec loss（传入 expert 因为 per-cluster）"""
        if expert is None:
            raise ValueError("[error] observe needs expert for per-cluster training")
        
        base_dist = expert.base_policy.forward(data)
        base_loss = expert.base_policy.compute_loss(data)  # 原 BC loss
        pred_action = base_dist.sample() # (B*T, ac_dim)
        batch_size = pred_action.shape[0] // self.n_tasks
        pred_action = pred_action.view(batch_size, self.n_tasks, pred_action.shape[-1])
        dec_loss = F.mse_loss(pred_action, data["actions"])

        loss = base_loss + self.cfg.policy.decorator_weight * dec_loss
        return loss

    def eval_observe(self, data, expert=None):
        """eval loss"""
        if expert is None:
            raise ValueError("[error] eval_observe needs expert")
        with torch.no_grad():
            return self.observe(data, expert)  # 复用，但 no_grad 在外部调用

    def build_skill_library(self, labeled_datasets, task_embs, benchmark, wandb_run=None):
        """Modified: Use pre-labeled datasets (with cluster_id / hyper)"""
        n_manip = len(labeled_datasets)
        device = self.cfg.device

        # 1. Group by cluster_id (from datasets)
        cluster_to_tasks = defaultdict(list)
        for i, ds in enumerate(labeled_datasets):
            cid = ds.cluster_id  # each ds has its cluster_id
            cluster_to_tasks[cid].append(i)

        # 2. Display clustering result
        print("\n=== Clustering Result (LIBERO tasks) ===")
        task_names = benchmark.get_task_names()
        for c in sorted(cluster_to_tasks.keys()):
            tasks = cluster_to_tasks[c]
            names = [task_names[t] for t in tasks]
            print(f"Cluster {c:2d} ({len(tasks)} tasks): {tasks} → {names}")

        # 3. Select representatives (using task_embs)
        features = torch.stack(tuple(task_embs)).cpu().numpy()
        print("\n=== Representatives ===")
        for c, tasks in cluster_to_tasks.items():
            feats = features[tasks]
            center = feats.mean(axis=0)
            dists = np.linalg.norm(feats - center, axis=1)
            rep_idx = tasks[np.argmin(dists)]
            print(f"Cluster {c:2d} representative: task {rep_idx} - {task_names[rep_idx]}")

        # 4. Train per-cluster ActionExpert (unchanged from previous)
        experts_best = {}
        per_task_best_results = {}

        for c, task_list in sorted(cluster_to_tasks.items()):
            print(f"\n=== Training ActionExpert for Cluster {c} ({len(task_list)} tasks) ===")

            expert = ActionExpert(self.cfg, self.cfg.shape_meta)
            expert.cluster_id = c
            expert = safe_device(expert, device)

            optimizer = eval(self.cfg.train.optimizer.name)(
                expert.parameters(), **self.cfg.train.optimizer.kwargs
            )

            # Prepare concat_ds from labeled_datasets in this cluster
            cluster_ds_list = [labeled_datasets[tid] for tid in task_list]
            concat_ds = ConcatDataset(cluster_ds_list)
            train_loader = DataLoader(
                concat_ds,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers,
                sampler=RandomSampler(concat_ds),
                persistent_workers=self.cfg.train.num_workers > 0,
            )
            print(f"Total data: {len(train_loader)}")

            best_succ = -1.0
            best_state = None
            best_epoch = 0

            for epoch in range(self.cfg.lifelong.n_epochs + 1):
                expert.train()
                total_loss = 0.0

                for idx, batch in enumerate(train_loader):
                    if idx >= 4: # FIXME
                        print(f"[DEBUG] early stop for testing.")
                        break
                    batch = self.map_tensor_to_device(batch)
                    optimizer.zero_grad()
                    loss = self.observe(batch, expert=expert)
                    (self.loss_scale * loss).backward()
                    if self.cfg.train.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(expert.parameters(), self.cfg.train.grad_clip)
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"  Epoch {epoch:3d} | loss {avg_loss:.4f}")

                if epoch % self.cfg.eval.eval_every == 0 or epoch == self.cfg.lifelong.n_epochs:
                    expert.eval()
                    
                    class TempAlgo:
                        def __init__(self, p, parent):
                            self.policy = p
                            self.cfg = parent.cfg
                            self.map_tensor_to_device = parent.map_tensor_to_device
                            self.eval = lambda: self.policy.eval()  # 确保 eval 模式
                            self.reset = lambda: self.policy.reset()  # 如果需要 reset

                    temp_algo = TempAlgo(expert, self)

                    succ_rates = []
                    for tid in task_list:
                        task = benchmark.get_task(tid)
                        task_emb = task_embs[tid]
                        # print(f"[DEBUG] eval at {tid}: {task.name}")
                        if self.cfg.eval.eval:
                            s = evaluate_one_task_success(
                                cfg=self.cfg,
                                algo=temp_algo,
                                task=task,
                                task_emb=task_emb,
                                task_id=tid,
                                sim_states=None,
                                task_str="",
                            )
                        else:
                            s = 0.0
                        print(f"Evaluate task {tid}: {s}")
                        succ_rates.append(s)
                    mean_succ = np.mean(succ_rates)
                    print(f"  Epoch {epoch:3d} | cluster validation succ {mean_succ:.4f} ± {np.std(succ_rates):.4f}")

                    if mean_succ > best_succ:
                        best_succ = mean_succ
                        best_state = expert.state_dict().copy()
                        best_epoch = epoch

                    if wandb_run is not None:
                        log_dict = {
                            f"cluster_{c}/epoch": epoch,
                            f"cluster_{c}/train_loss": avg_loss,
                            f"cluster_{c}/val_succ": mean_succ,
                            f"cluster_{c}/val_succ_std": np.std(succ_rates),
                            f"cluster_{c}/best_succ": best_succ,
                        }
                        for idx, tid in enumerate(task_list):
                            log_dict[f"cluster_{c}/task_{tid}_succ"] = succ_rates[idx]
                        wandb_run.log(log_dict)

            save_path = os.path.join(self.experiment_dir, f"action_expert_cluster_{c}_best.pth")
            torch_save_model(expert, save_path, cfg=self.cfg)
            print(f"Model save to {save_path}.")

            experts_best[c] = (best_state, best_succ, best_epoch)

            for tid in task_list:
                per_task_best_results[tid] = (best_succ, c)

            if wandb_run is not None:
                wandb_run.log({f"final/cluster_{c}_best_succ": best_succ})
            # else:
            #     print(f"=== Per-task Success ===")
            #     for tid in task_list:
            #         print(benchmark.get_task(tid).name, end=': ')
            #         print(per_task_best_results[tid])


        # 5. Final output + log (unchanged)
        print("\n=== Final Skill Library Results (best validation per expert) ===")
        for tid in sorted(per_task_best_results.keys()):
            succ, c = per_task_best_results[tid]
            print(f"Task {tid:2d} ({task_names[tid]}) → Cluster {c} | best succ {succ:.4f}")
            if wandb_run is not None:
                wandb_run.log({f"final/task_{tid}_best_succ": succ, f"final/task_{tid}_cluster": c})

        print(f"\n[info] Skill Library built! {len(experts_best)} experts saved in {self.experiment_dir}")
        return experts_best