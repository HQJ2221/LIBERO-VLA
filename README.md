# LIBERO-VLA

a developing experiment based on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

- LIBERO usage please refer to [README_LIBERO.md](README_LIBERO.md).

## Some usual commands

- Download datasets (where `DATASET` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal]`):

```bash
python benchmark_scripts/download_libero_datasets.py --datasets DATASET --use-huggingface
```

- Train a lifelong learning policy:
  - `BENCHMARK` from `[LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_90, LIBERO_10]`
  - `POLICY` from `[bc_rnn_policy, bc_transformer_policy, bc_vilt_policy, bc_action_expert]`
  - `LIFELONG` from `[base, er, ewc, packnet, multitask, skill_library]`

```bash
CUDA_VISIBLE_DEVICES=GPU_ID MUJOCO_EGL_DEVICE_ID=GPU_ID \
python libero/lifelong/main.py seed=SEED \
    benchmark_name=BENCHMARK \
    policy=POLICY \
    lifelong=LIFELONG \
```

- We add `main_skill.py` for our experiment. (Refer to `./run.sh`)