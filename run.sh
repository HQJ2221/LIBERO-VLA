CUDA_VISIBLE_DEVICES=0,1 MUJOCO_EGL_DEVICE_ID=0 PYTHONWARNINGS="ignore" \
python libero/lifelong/main_skill.py seed=10000 \
    benchmark_name=LIBERO_GOAL \
    policy=bc_action_expert \
    lifelong=skill_library \