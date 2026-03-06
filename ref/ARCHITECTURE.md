# 6-DOF 机械臂视觉抓取系统 — 整体架构设计

> 最后更新: 2026-03-03

---

## 0. 项目进度总览

### 已完成

| # | 任务 | 日期 | 说明 |
|---|------|------|------|
| 1 | 论文调研 | 2026-03 | 下载 42 篇最新论文 (2021-2026), 覆盖 VLA/RL/Grasp Detection |
| 2 | FK/IK 运动学模块 | 2026-03 | `robot_arm/kinematics/`, URDF+DH 双模式, 9 项测试全通过 |
| 3 | MuJoCo 验证 | 2026-03 | FK 精度 3.4e-16m, Jacobian 1.67e-15, 轨迹跟踪 sub-0.1mm |
| 4 | MuJoCo 渲染演示 | 2026-03 | 6 个 demo 视频 + 关键帧图, 含灯光/地面/阴影 |
| 5 | 系统架构设计 v1 | 2026-03 | 分层几何抓取方案 (已被 v2 替代) |
| 6 | 系统架构设计 v2 | 2026-03 | **Isaac Lab RL + VLA 基础模型** 混合方案 (当前) |

### 进行中

| # | 任务 | 状态 | 说明 |
|---|------|------|------|
| 7 | Isaac Lab 环境搭建 | **待开始** | 安装 Isaac Sim + Isaac Lab, 导入 URDF |

### 待做 (TODO)

详见 **[第 6 节: 实施路线图](#6-实施路线图-todo-list)**.

---

## 1. 硬件与环境

### 1.1 开发机配置

| 项目 | 规格 |
|------|------|
| CPU | AMD Ryzen AI 7 H 350 w/ Radeon 860M |
| GPU | **NVIDIA GeForce RTX 5060 Laptop, 8GB VRAM** |
| RAM | 31.1 GB DDR5 |
| 磁盘 | D: 3.7TB (可用 2.4TB) |
| OS | Windows 11 Home China (26200) |
| Python | 3.13.5 (base conda) |
| PyTorch | 2.10.0+cu128 (CUDA 12.8) |
| MuJoCo | 3.5.0 |
| Conda | 25.7.0 |

### 1.2 机械臂硬件

| 组件 | 型号 | 说明 |
|------|------|------|
| 机械臂 | 自研 6-DOF | Robstride 电机 (04/03/00 型), CAN 总线通信 |
| 夹爪 | 平行夹爪 | 第 7 轴 (Robstride 05), 开合控制 |
| 深度相机 | Intel RealSense D435i (待采购确认) | RGB 1920x1080 + 深度 1280x720, 30fps |
| 通信 | USB-CAN (ControlCAN.dll) | MIT 模式: 位置+速度+力矩 @ 200Hz |

### 1.3 已有软件模块

| 模块 | 位置 | 状态 | 功能 |
|------|------|------|------|
| 运动学 | `robot_arm/kinematics/` | **已完成** | FK/IK/Jacobian, URDF+DH 双模式, MuJoCo 验证通过 |
| 动力学 | `grpc_stream/Inv_Dyn_2.py` | 已有 | Newton-Euler 逆动力学 (6 DOF 重力补偿) |
| 电机驱动 | `grpc_stream/robstride.py` | 已有 | CAN 总线 MIT/POS 模式控制 |
| 遥操作 | `grpc_stream/server.py + client.py` | 已有 | gRPC 双向流, 主从手 @ 200Hz |
| 轨迹插值 | `grpc_stream/Interpolation.py` | 已有 | 三次样条插值 |
| MuJoCo 仿真 | `robot_arm/verification/` | **已完成** | 模型加载、FK/IK 验证、渲染演示 |

### 1.4 URDF 模型

```
arm/urdf/arm.urdf          — SolidWorks 导出, 6 旋转关节 + 2 夹爪体
arm/meshes/0-6.STL, 7-1.STL, 7-2.STL  — 9 个 STL mesh
```

---

## 2. 架构总览 (2026 最新方案)

基于 42 篇论文 + 2026 最新调研，采用 **Isaac Lab RL → VLA 基础模型** 渐进路线。

### 2.1 为什么选这条路

| 对比项 | 传统分层 (E3GNet) | **Isaac Lab RL** (选择) | 端到端 VLA |
|--------|-------------------|------------------------|-----------|
| 真实数据需求 | 0 | **0** | 50-100 demo |
| 训练方式 | 不需训练 | **仿真 RL 自动探索** | 监督学习 |
| 端到端 | 否 (多模块) | **是 (单策略)** | 是 |
| Sim-to-Real | 分层各自迁移 | **统一域随机化** | 需真实数据 |
| 语言指令 | 需额外模块 | 不支持 | **原生支持** |
| 泛化能力 | 物体几何泛化 | 域随机化范围内 | **开放世界** |
| 可扩展性 | 难 | 好 (→ VLA 蒸馏) | **最好** |
| 适合当前阶段 | 可以 | **最佳** | 缺数据 |

**核心逻辑:**
1. Isaac Lab RL 零数据启动, 仿真训练, URDF 直接导入
2. 训好的 RL 策略可辅助采集遥操作数据
3. 有数据后升级到 GR00T N1.6 / π₀.5 获得语言+泛化能力

### 2.2 四阶段路线图

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Phase 1: Isaac Lab RL 仿真训练         ← 当前阶段       │
│  ─────────────────────────────────                       │
│  URDF → Isaac Lab → PPO 训练抓取策略                      │
│  域随机化 → Sim-to-Real 部署                              │
│  目标: 真机能抓桌上物体 (>90%)                             │
│                                                          │
│  Phase 2: 感知增强 + 真机部署                              │
│  ───────────────────────────                             │
│  RealSense 集成 + 手眼标定                                │
│  RL 策略 + 视觉观测 (RGB-D)                               │
│  目标: 视觉引导抓取, 处理未知物体                           │
│                                                          │
│  Phase 3: 数据采集 + VLA 微调                              │
│  ─────────────────────────                               │
│  RL 策略辅助遥操作, 采集 50-100 demo                       │
│  微调 GR00T N1.6 或 π₀.5 (OpenPI)                        │
│  目标: 语言指令抓取 ("拿起红色杯子")                        │
│                                                          │
│  Phase 4: RL + VLA 混合优化 (2026 SOTA)                   │
│  ──────────────────────────────                          │
│  iRe-VLA: 交替 RL 在线优化 + SFT                          │
│  目标: 开放世界泛化 + 持续自我改进                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: Isaac Lab RL 训练 (详细设计)

### 3.1 Isaac Lab 技术栈

```
Isaac Sim 5.1 (物理仿真引擎, PhysX 5)
    │
    └── Isaac Lab 2.3 (机器人学习框架)
            │
            ├── URDF Importer  ← arm/urdf/arm.urdf
            ├── Environment    ← 自定义 Pick-and-Place Env
            ├── RL Training    ← PPO (RSL-RL / RL-Games / SKRL)
            ├── Domain Rand.   ← ADR (自动域随机化)
            └── Sim-to-Real    ← 策略导出 → 真机部署
```

### 3.2 系统流程

```
                     ┌─ Isaac Lab 仿真 ──────────────────┐
                     │                                    │
arm.urdf ──→ [URDF Import] ──→ USD 资产                   │
                     │                                    │
                     │  ┌───── RL 训练循环 ─────────┐     │
                     │  │                           │     │
                     │  │  观测 obs(t)              │     │
                     │  │  ├─ 关节角 q (6,)          │     │
                     │  │  ├─ 关节角速度 dq (6,)     │     │
                     │  │  ├─ 末端位姿 ee_pos (3,)   │     │
                     │  │  ├─ 物体位姿 obj_pos (3,)  │     │
                     │  │  ├─ 目标位姿 goal_pos (3,) │     │
                     │  │  └─ 夹爪状态 gripper (1,)  │     │
                     │  │         │                  │     │
                     │  │         ▼                  │     │
                     │  │  [策略网络 π(obs)]          │     │
                     │  │  MLP: 256-256-128          │     │
                     │  │         │                  │     │
                     │  │         ▼                  │     │
                     │  │  动作 act(t)               │     │
                     │  │  ├─ 关节位置增量 Δq (6,)   │     │
                     │  │  └─ 夹爪命令 (1,)          │     │
                     │  │         │                  │     │
                     │  │         ▼                  │     │
                     │  │  [奖励 r(t)]               │     │
                     │  │  ├─ 接近奖励               │     │
                     │  │  ├─ 抓取奖励               │     │
                     │  │  ├─ 提升奖励               │     │
                     │  │  └─ 成功奖励               │     │
                     │  │         │                  │     │
                     │  │         ▼                  │     │
                     │  │  [PPO 更新]                │     │
                     │  │  16384 并行环境             │     │
                     │  └───────────────────────────┘     │
                     └────────────────────────────────────┘
                                  │
                                  ▼ 导出策略
                     ┌─ 真机部署 ────────────────────────┐
                     │                                    │
                     │  策略网络 π(obs)                    │
                     │       │                            │
                     │       ▼                            │
                     │  Δq → q_target = q_current + Δq    │
                     │       │                            │
                     │       ▼                            │
                     │  逆动力学 → 力矩前馈               │
                     │       │                            │
                     │       ▼                            │
                     │  MIT 控制 → CAN → Robstride 电机   │
                     └────────────────────────────────────┘
```

### 3.3 奖励函数设计 (Pick-and-Place)

```python
reward = (
    # Stage 1: 末端接近物体
    + w1 * exp(-k1 * ||ee_pos - obj_pos||)

    # Stage 2: 夹爪对准
    + w2 * exp(-k2 * align_error)

    # Stage 3: 抓取成功 (物体与夹爪接触)
    + w3 * grasp_contact_reward

    # Stage 4: 提升物体
    + w4 * max(0, obj_height - table_height)

    # Stage 5: 到达目标
    + w5 * exp(-k5 * ||obj_pos - goal_pos||)

    # 惩罚
    - w6 * ||action||^2            # 动作平滑
    - w7 * joint_limit_penalty     # 关节限位
    - w8 * collision_penalty       # 自碰撞
)
```

### 3.4 域随机化 (Sim-to-Real)

| 参数 | 随机化范围 | 说明 |
|------|-----------|------|
| 物体质量 | 0.01 - 2.0 kg | 覆盖日常物品 |
| 物体摩擦 | 0.3 - 1.0 | 光滑到粗糙 |
| 物体形状 | 方块/圆柱/球 + YCB | 基础形状+真实物体 |
| 物体位置 | 桌面随机 | 工作空间内均匀 |
| 关节摩擦 | ±30% | 电机特性差异 |
| 观测噪声 | ±0.01 rad | 编码器噪声 |
| 控制延迟 | 0-20 ms | 通信延迟 |
| 重力偏差 | ±0.05 m/s² | 安装倾斜 |

### 3.5 文件结构

```
robot_arm/
├── isaac_lab/                    # [新建] Isaac Lab 训练环境
│   ├── __init__.py
│   ├── assets/
│   │   ├── arm.usd              # URDF→USD 转换后的资产
│   │   └── objects/             # 抓取目标物体 USD
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── grasp_env.py         # 抓取训练环境 (Manager-based)
│   │   └── grasp_env_cfg.py     # 环境配置
│   ├── mdp/
│   │   ├── __init__.py
│   │   ├── observations.py      # 观测定义
│   │   ├── rewards.py           # 奖励函数
│   │   ├── terminations.py      # 终止条件
│   │   └── actions.py           # 动作空间
│   ├── train.py                 # PPO 训练脚本
│   ├── play.py                  # 策略评估/可视化
│   └── deploy.py                # 真机部署
│
├── kinematics/                  # [已完成]
├── verification/                # [已完成]
└── grpc_stream/                 # [已有]
```

---

## 4. Phase 2-4 概要

### Phase 2: 感知增强 + 真机部署

```
新增:
├── perception/
│   ├── camera.py            # RealSense D435i 驱动
│   ├── point_cloud.py       # 深度图→点云
│   └── hand_eye_calib.py    # 手眼标定
├── control/
│   ├── robot_interface.py   # 统一硬件接口 (CAN/Sim 切换)
│   └── safety.py            # 安全限制
```

### Phase 3: VLA 基础模型微调

```
方案 A: GR00T N1.6 (NVIDIA, 开源)
  - 已有 LeRobot SO-101 单臂微调教程
  - 部署到 Jetson AGX Orin
  - Diffusion Transformer 动作头

方案 B: π₀.5 / OpenPI (Physical Intelligence, 开源)
  - 需改 action dim 从 14→7 (6 关节 + 夹爪)
  - JAX 训练, 1-20 小时数据即可微调
  - Flow matching 动作生成

新增:
├── learning/
│   ├── data_collector.py    # 遥操作数据采集 (LeRobot v2.1 格式)
│   ├── groot_finetune.py    # GR00T N1.6 微调
│   └── deploy_vla.py        # VLA 策略部署
```

### Phase 4: RL + VLA 混合优化

```
iRe-VLA 方法: 交替 RL + SFT 迭代
├── RL 在线优化: Isaac Lab 中用 VLA 策略做 RL fine-tuning
├── SFT 扩增: 成功轨迹回流到 VLA 训练集
└── 持续进化: VLA 策略越来越强
```

---

## 5. 环境需求 & 安装计划

### 5.1 Isaac Sim + Isaac Lab 要求

| 需求 | 当前 | 状态 |
|------|------|------|
| NVIDIA GPU (RTX) | RTX 5060 Laptop 8GB | **OK** (最低 8GB) |
| GPU 驱动 | 577.05 | **OK** |
| CUDA 12.x | 12.8 (via PyTorch) | **OK** |
| RAM | 31.1 GB | **OK** (推荐 32GB) |
| 磁盘空间 | 2.4 TB 可用 | **OK** (需 ~50GB) |
| OS | Windows 11 | **OK** (Isaac Sim 支持) |
| Python | 3.13.5 | **需新建 conda env (3.10/3.11)** |

### 5.2 安装步骤 (TODO)

```
Step 1: 安装 Isaac Sim 5.1
  ├── 方式 A: Omniverse Launcher (GUI)
  │   └── 下载: https://developer.nvidia.com/isaac-sim
  └── 方式 B: pip install (实验性)
      └── pip install isaacsim==5.1.0

Step 2: 新建 Conda 环境
  └── conda create -n isaaclab python=3.11
      conda activate isaaclab

Step 3: 安装 Isaac Lab 2.3
  ├── git clone https://github.com/isaac-sim/IsaacLab.git
  ├── cd IsaacLab
  └── ./isaaclab.sh --install  (或 Windows: isaaclab.bat --install)

Step 4: 验证安装
  └── python -c "import isaaclab; print(isaaclab.__version__)"

Step 5: 导入 URDF
  └── python scripts/tools/convert_urdf.py arm/urdf/arm.urdf arm.usd

Step 6: 创建自定义抓取环境
  └── 参考 isaacLab.manipulation 模板
```

### 5.3 备选方案 (如果 Isaac Lab 安装困难)

Isaac Sim 体量大 (~50GB), 对 Windows laptop 可能有挑战。备选:

| 方案 | 优势 | 劣势 |
|------|------|------|
| **MuJoCo + Gymnasium** | 轻量, 已安装, 易上手 | 并行度低, 无 PhysX |
| **ManiSkill3** | GPU 并行, 支持 URDF | 社区较小 |
| **Isaac Gym (旧版)** | 成熟稳定 | 已停止更新 |
| **云端 Isaac Lab** | 无本地限制 | 需云 GPU, 有成本 |

如果 Isaac Lab 安装不顺，**MuJoCo + Gymnasium + Stable-Baselines3** 是最轻量的替代，已有 MuJoCo 基础可直接复用。

---

## 6. 实施路线图 (TODO List)

### Phase 1: Isaac Lab RL 训练

#### Step 1: 环境搭建 ⬜

- [ ] **1.1** 安装 Isaac Sim 5.1 (Omniverse Launcher 或 pip)
- [ ] **1.2** 创建 conda env `isaaclab` (Python 3.11)
- [ ] **1.3** 安装 Isaac Lab 2.3
- [ ] **1.4** 运行官方 demo 验证 (`Franka Reach` / `Franka Lift-Cube`)
- [ ] **1.5** 将 `arm/urdf/arm.urdf` 导入为 USD 资产
- [ ] **1.6** 在 Isaac Lab 中可视化机械臂, 验证关节运动正确

#### Step 2: 自定义抓取环境 ⬜

- [ ] **2.1** 创建 `robot_arm/isaac_lab/` 目录结构
- [ ] **2.2** 定义 `grasp_env_cfg.py` — 场景 (桌+臂+物体)
- [ ] **2.3** 定义观测空间 (关节角/物体位姿/目标)
- [ ] **2.4** 定义动作空间 (关节增量 + 夹爪)
- [ ] **2.5** 实现奖励函数 (reach → grasp → lift)
- [ ] **2.6** 实现终止条件 (超时/成功/掉落)
- [ ] **2.7** 加入域随机化 (物体形状/质量/摩擦/位置)
- [ ] **2.8** 测试: 随机策略能跑通 env, 无报错

#### Step 3: RL 训练 ⬜

- [ ] **3.1** 选择 RL 库 (RSL-RL / SKRL / RL-Games)
- [ ] **3.2** 配置 PPO 超参数
- [ ] **3.3** 小规模训练测试 (256 envs, 快速验证)
- [ ] **3.4** 正式训练 (4096+ envs, 观察收敛曲线)
- [ ] **3.5** 策略评估: 抓取成功率 > 90%
- [ ] **3.6** 消融实验: 不同奖励权重/网络结构对比

#### Step 4: Sim-to-Real 部署 ⬜

- [ ] **4.1** 导出训练好的策略为 ONNX/TorchScript
- [ ] **4.2** 编写真机推理脚本 (`deploy.py`)
- [ ] **4.3** 策略输出 → 关节目标 → Inv_Dyn → MIT 控制
- [ ] **4.4** 先在 MuJoCo 验证 (非 Isaac Lab 环境)
- [ ] **4.5** 真机测试: 简单物体抓取

### Phase 2: 感知 + 真机 ⬜

- [ ] **5.1** 安装 pyrealsense2, 验证相机数据流
- [ ] **5.2** 手眼标定 (eye-to-hand)
- [ ] **5.3** RL 策略加入视觉观测 (RGB / Depth)
- [ ] **5.4** 视觉策略仿真训练
- [ ] **5.5** 视觉策略真机测试

### Phase 3: VLA 微调 ⬜

- [ ] **6.1** 用 RL 策略辅助遥操作, 采集 50+ demo
- [ ] **6.2** 数据转 LeRobot v2.1 格式
- [ ] **6.3** 微调 GR00T N1.6 或 π₀.5
- [ ] **6.4** VLA 策略部署测试

---

## 7. 技术决策记录

### Q1: 为什么选 Isaac Lab 而不是直接几何抓取?

2026 年业界共识: **RL + Foundation Model 混合** 是最佳路径
(参考: [Embodied Robot Manipulation in the Era of Foundation Models](https://arxiv.org/abs/2512.22983))
- Isaac Lab 零数据启动, GPU 并行训练 (160万 FPS)
- 端到端单策略, 不需要分层维护
- 训好的 RL 策略是 VLA 微调的最佳数据引擎

### Q2: RTX 5060 Laptop (8GB VRAM) 够用吗?

- Isaac Lab 训练: **够用** (4096 envs 约需 4-6GB)
- 不够时: 减少并行环境数 (2048/1024)
- Isaac Sim 渲染: 可能吃紧, 训练时关闭渲染 (headless)
- VLA 推理: 小模型 (~1B) 可以, 大模型需量化或云端

### Q3: Windows 能跑 Isaac Lab 吗?

- Isaac Sim 5.x 官方支持 Windows
- Isaac Lab 2.3 支持 Windows (通过 `isaaclab.bat`)
- 已知限制: 部分 feature 在 Linux 更稳定
- 备选: WSL2 + NVIDIA GPU passthrough

### Q4: 为什么 Phase 1 不用视觉?

- 先用 state-based RL (关节角+物体位姿) 验证训练流程
- State-based 训练快 (几小时), 调试方便
- 视觉观测 (image-based) 训练慢很多, 放到 Phase 2
- 这是 Isaac Lab 官方推荐的渐进路线

### Q5: GR00T N1.6 vs π₀.5 怎么选?

| | GR00T N1.6 | π₀.5 (OpenPI) |
|---|---|---|
| 开发商 | NVIDIA | Physical Intelligence |
| 生态 | Isaac Lab 原生集成 | 独立, LeRobot 兼容 |
| 单臂支持 | 已验证 (SO-101) | 需改 action dim |
| 推理硬件 | Jetson AGX Orin | GPU (JAX) |
| 训练框架 | PyTorch | JAX |
| 推荐 | **首选** (与 Isaac Lab 生态一致) | 备选 |

---

## 8. 参考资料

### 核心工具
- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab) — 主框架
- [isaacLab.manipulation](https://github.com/NathanWu7/isaacLab.manipulation) — 抓取环境模板
- [Isaac Lab Arena](https://developer.nvidia.com/isaac/lab-arena) — 策略评估框架
- [GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA VLA 基础模型
- [OpenPI (π₀)](https://github.com/Physical-Intelligence/openpi) — Physical Intelligence VLA
- [LeRobot](https://github.com/huggingface/lerobot) — HuggingFace 机器人学习

### 关键论文
- Isaac Lab: GPU-Accelerated Simulation Framework (arXiv:2511.04831)
- π₀: Vision-Language-Action Flow Model (arXiv:2410.24164)
- π₀.5: VLA with Open-World Generalization (arXiv:2504.16054)
- Embodied Robot Manipulation in the Era of Foundation Models (arXiv:2512.22983)
- Diffusion Policy (arXiv:2303.04137)
- ACT: Action Chunking with Transformers (arXiv:2304.13705)

### 本项目已下载论文 (42篇)
存放于 `robotarm_ws/papers/` 和根目录 PDF 文件。
