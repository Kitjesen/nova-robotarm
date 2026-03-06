# Isaac Lab 安装与环境搭建笔记

> 调研日期: 2026-03-03

## 1. 版本信息

- Isaac Sim: **5.1.0** (2025-10-21)
- Isaac Lab: **2.3.2** (2026-02-02, 基于 Isaac Sim 5.1)
- Python: **必须 3.11** (3.13 不兼容)
- PyTorch: **2.7.0+cu128**
- NVIDIA 驱动: **必须 ≥580.88** (当前 577.05 需升级)

## 2. 安装步骤

```bash
# 0. 前置: 升级 NVIDIA 驱动到 580.88+
# 0. 前置: 启用 Windows 长路径 (管理员 PowerShell):
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# 1. 创建 conda 环境
conda create -n isaaclab python=3.11 -y
conda activate isaaclab

# 2. 安装 PyTorch
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 3. 安装 Isaac Sim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 4. 克隆 Isaac Lab
cd D:\inovxio\products\robotarm_ws
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 5. 安装 Isaac Lab
isaaclab.bat --install

# 6. 验证
isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
```

## 3. RTX 5060 Laptop (8GB VRAM) 注意事项

- 官方最低 16GB VRAM, 8GB 低于要求
- **必须用 --headless 模式训练** (不开渲染)
- 并行环境数建议 512-2048 (而非标准 4096)
- 已知问题: GitHub #3466 (Isaac Sim 5.0 在 5060 上崩溃), 用 5.1 应该更好

## 4. URDF 导入

```bash
isaaclab.bat -p scripts/tools/convert_urdf.py arm/urdf/arm.urdf arm/usd/arm.usd --merge-joints --fix-base
```

关键参数:
- fix_base=True (固定基座)
- make_instanceable=True (并行环境)
- merge_fixed_joints=True
- collision_from_visuals=False

## 5. 自定义环境模板

参考 Isaac Lab 内置 Franka Lift-Cube:
```
isaaclab_tasks/manager_based/manipulation/lift/
├── lift_env_cfg.py      # 场景+奖励+终止+事件
├── mdp/                 # 自定义 MDP 函数
└── config/franka/       # Franka 特定配置
```

关键环境参数:
- num_envs: 4096 (我们可能需降到 512-2048)
- decimation: 2 (控制 50Hz, 物理 100Hz)
- episode_length_s: 5.0
- sim.dt: 0.01

## 6. RL 训练

推荐: **RSL-RL + PPO** (最快, 最常用于 Isaac Lab manipulation)

训练命令:
```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-v0 --num_envs 2048 --headless
```

PPO 关键超参:
- learning_rate: 1e-3 (adaptive schedule)
- gamma: 0.99
- lam: 0.95 (GAE)
- num_learning_epochs: 5
- num_mini_batches: 4
- clip_param: 0.2
- actor_hidden_dims: [256, 128, 64]

## 7. 奖励设计 (Franka Lift-Cube 参考)

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| reaching_object | 1.0 | tanh(ee-obj距离), std=0.1 |
| lifting_object | **15.0** | 物体高度>4cm |
| object_goal_tracking | **16.0** | 粗粒度目标跟踪, std=0.3 |
| object_goal_tracking_fine | 5.0 | 细粒度跟踪, std=0.05 |
| action_rate | -1e-4 | 动作平滑惩罚 |
| joint_vel | -1e-4 | 关节速度惩罚 |

关键: lifting 权重远大于 reaching, 防止 agent 只悬停不抓
