import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import multiprocessing
import pprint
import time
from pathlib import Path

import hydra
import numpy as np
import wandb
import yaml
import torch
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class, get_algo_list
from libero.lifelong.models import get_policy_list
from libero.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, SkillLabeledVLDataset, get_dataset
from libero.lifelong.metric import evaluate_loss, evaluate_success
from libero.lifelong.utils import (
    NpEncoder,
    compute_flops,
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
    torch_save_model,
)


@hydra.main(config_path="../configs", config_name="config_skill", version_base=None)
def main(hydra_cfg):
    # FIXME: 配置eval的渲染env变量
    # os.environ['MUJOCO_GL'] = 'egl'
    # os.environ['PYOPENGL_PLATFORM'] = 'egl'
    print("[DEBUG] MUJOCO_GL:", os.environ.get('MUJOCO_GL', 'Not set'))

    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    pp.pprint("Available algorithms:")
    pp.pprint(get_algo_list())

    pp.pprint("Available policies:")
    pp.pprint(get_policy_list())

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks

    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    shape_meta = None

    for i in range(n_manip_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
            )
        except Exception as e:
            print(
                f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}"
            )
            print(f"[error] {e}")
        print(os.path.join(cfg.folder, benchmark.get_task_demonstration(i)))
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)

    print(f"[INFO] Getting task embed...")
    start = time.time()
    task_embs = get_task_embs(cfg, descriptions)  # FIXME: this can be modified to optimize the state construction
    print(f"[INFO] Get task embed using {time.time() - start}s.")
    benchmark.set_task_embs(task_embs)

    # prepare experiment and update the config
    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta

    ########################################################
    #    Load or Train cluster model
    ########################################################
    from libero.lifelong.models.cluster_model import ClusterModel  # 导入你新定义的类

    device = cfg.device
    embed_dim = task_embs[0].shape[0]  # 假设所有 emb dim 相同
    num_clusters = cfg.num_clusters
    hyper_dim = cfg.hyper_dim

    if not cfg.pretrain_cluster:
        from torch import optim
        import torch.nn as nn
        
        print("[info] No pretrained cluster model, training new one...")
        cluster_model = ClusterModel(embed_dim, num_clusters, hyper_dim)
        cluster_model = safe_device(cluster_model, device)
        
        # 训练 ClusterModel（用 task_embs as data；无需 full dataset）
        optimizer = optim.Adam(cluster_model.parameters(), lr=cfg.cluster_lr)
        task_embs_tensor = torch.stack(tuple(task_embs), dim=0).to(device)  # (n_tasks, embed_dim)
        
        for epoch in range(cfg.cluster_epochs):
            epoch_loss = 0.0
            for step in range (cfg.cluster_steps_per_epoch):
                cluster_id, hyper, latent, dists = cluster_model(task_embs_tensor)

                hard_loss = torch.mean(torch.gather(dists, 1, cluster_id.unsqueeze(1)))   # k-means 目标

                # entropy 正则化（防止全部 collapse）
                probs = torch.softmax(-dists / cluster_model.temperature, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

                loss = hard_loss - 0.02 * entropy          # 0.02 是推荐系数，可在 cfg 里调

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % 10 == 0:
                print(f"Cluster train epoch {epoch}, loss: {(epoch_loss / cfg.cluster_steps_per_epoch):.4f}")
        
        # 保存新训模型
        cluster_model_path = cfg.cluster_model_path or os.path.join(cfg.experiment_dir, "cluster_model.pth")
        torch_save_model(cluster_model, cluster_model_path, cfg=cfg)
        print(f"[info] Trained cluster model saved to {cluster_model_path}")
    else:
        assert os.path.exists(cfg.cluster_model_path), f"[error] pretrained cluster model path {cfg.cluster_model_path} does not exist"
        print(f"[info] Loading pretrained cluster model from {cfg.cluster_model_path}")
        cluster_model = ClusterModel(embed_dim, num_clusters, hyper_dim)
        cluster_model = safe_device(cluster_model, device)
        cluster_model.load_state_dict(torch_load_model(cfg.cluster_model_path)[0])
        cluster_model.eval()

    # 用 cluster_model 为每个 task 计算 cluster_id / hyper
    task_embs_tensor = torch.stack(tuple(task_embs), dim=0).to(device)
    with torch.no_grad():
        cluster_ids, hypers, _, _ = cluster_model(task_embs_tensor)  # (n_tasks,), (n_tasks, hyper_dim)
    
    ##########################
    #       Wandb Init       #
    ##########################
    if cfg.use_wandb:
        if "run_name" not in cfg or cfg.run_name is None:
            cfg.run_name = f"{benchmark.name}_{cfg.lifelong.algo}_{int(time.time())}"
        if "run_id" not in cfg or cfg.run_id is None:
            cfg.run_id = cfg.run_name
        wandb.init(id=cfg.run_id, name=cfg.run_name, entity=cfg.wandb_entity, project=cfg.wandb_project, config=cfg)
        wandb_run = wandb.run
    else:
        wandb_run = None

    # Visualization of Clustering
    if cfg.use_wandb:
        # 1. 表格（保持不变）
        cluster_ids_np = cluster_ids.cpu().numpy()
        task_names = benchmark.get_task_names()

        table = wandb.Table(columns=["task_id", "task_name", "cluster_id"])
        for i in range(n_manip_tasks):
            table.add_data(i, task_names[i], int(cluster_ids_np[i]))
        wandb.log({"cluster_assignment_table": table})

        # 2. 用 matplotlib 画彩色 PCA 散点图（带 legend + 任务名）
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        pca = PCA(n_components=2)
        task_embs_2d = pca.fit_transform(task_embs_tensor.cpu().numpy())

        fig, ax = plt.subplots(figsize=(12, 9))
        colors = plt.cm.tab10(np.linspace(0, 1, cfg.num_clusters))

        for c in range(cfg.num_clusters):
            mask = (cluster_ids_np == c)
            ax.scatter(task_embs_2d[mask, 0], task_embs_2d[mask, 1],
                       c=[colors[c]], label=f"Cluster {c} ({mask.sum()} tasks)",
                       s=80, alpha=0.9)

        # 标注任务名（前 15 个字符避免重叠）
        for i in range(n_manip_tasks):
            ax.text(task_embs_2d[i, 0] + 0.02, task_embs_2d[i, 1],
                    task_names[i][:15], fontsize=7, alpha=0.7)

        ax.set_title("Task Embedding PCA Scatter with Clusters (Colored)", fontsize=14)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # 上传到 wandb（清晰矢量图）
        wandb.log({"task_embedding_pca_matplotlib": wandb.Image(fig)})
        plt.close(fig)   # 释放内存

    gsz = cfg.data.task_group_size
    if gsz == 1:  # each manipulation task is its own lifelong learning task
        datasets = []
        for i in range(n_manip_tasks): # Notice: change SequenceVLDataset to SkillLabeledVLDataset
            ds = SkillLabeledVLDataset(
                sequence_dataset=manip_datasets[i],
                task_emb=task_embs[i],
                cluster_id=cluster_ids[i].item(),
                skill_hyper_params=hypers[i],  # per-task hyper (tensor)
            )
            datasets.append(ds)
        n_demos = [data.n_demos for data in datasets]
        n_sequences = [data.total_num_sequences for data in datasets]
    else:  # group gsz manipulation tasks into a lifelong task, currently not used
        assert (
            n_manip_tasks % gsz == 0
        ), f"[error] task_group_size does not divide n_tasks"
        datasets = []
        n_demos = []
        n_sequences = []
        for i in range(0, n_manip_tasks, gsz):
            dataset = GroupedTaskDataset(
                manip_datasets[i : i + gsz], task_embs[i : i + gsz]
            )
            datasets.append(dataset)
            n_demos.extend([x.n_demos for x in dataset.sequence_datasets])
            n_sequences.extend(
                [x.total_num_sequences for x in dataset.sequence_datasets]
            )

    n_tasks = n_manip_tasks // gsz  # number of lifelong learning tasks
    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks // gsz}")
    # for i in range(n_tasks):
    #     print(f"    - Task {i+1}:")
    #     for j in range(gsz):
    #         print(f"        {benchmark.get_task(i*gsz+j).language}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    result_summary = {
        "L_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # loss confusion matrix
        "S_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # success confusion matrix
        "L_fwd": np.zeros((n_manip_tasks,)),  # loss AUC, how fast the agent learns
        "S_fwd": np.zeros((n_manip_tasks,)),  # success AUC, how fast the agent succeeds
    }

    if cfg.eval.save_sim_states:
        # for saving the evaluate simulation states, so we can replay them later
        for k in range(n_manip_tasks):
            for p in range(k + 1):  # for testing task p when the agent learns to task k
                result_summary[f"k{k}_p{p}"] = [[] for _ in range(cfg.eval.n_eval)]
            for e in range(
                cfg.train.n_epochs + 1
            ):  # for testing task k at the e-th epoch when the agent learns on task k
                if e % cfg.eval.eval_every == 0:
                    result_summary[f"k{k}_e{e//cfg.eval.eval_every}"] = [
                        [] for _ in range(cfg.eval.n_eval)
                    ]

    # define lifelong algorithm
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)
    if cfg.pretrain_model_path != "":  # load a pretrained model if there is any
        try:
            algo.policy.load_state_dict(torch_load_model(cfg.pretrain_model_path)[0])
        except:
            print(
                f"[error] cannot load pretrained model from {cfg.pretrain_model_path}"
            )
            sys.exit(0)

    print(f"[info] start lifelong learning with algo {cfg.lifelong.algo}")
    GFLOPs, MParams = compute_flops(algo, datasets[0], cfg)
    print(f"[info] policy has {GFLOPs:.1f} GFLOPs and {MParams:.1f} MParams\n")

    # save the experiment config file, so we can resume or replay later
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)


    # TODO: rewrite the skill library building process
    algo.build_skill_library(datasets, task_embs, benchmark, wandb_run=wandb_run)

    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Set the multiprocessing start method to 'fork'. 
    # 'spawn' may cause conflict between multiprocess and h5df object pickle.
    if multiprocessing.get_start_method(allow_none=True) != "fork":  
        multiprocessing.set_start_method("fork", force=True)
    main()
