# 机械臂视觉抓取 — 完整技术路线图

**最后更新：2026-03-06**
**当前状态：Phase 1 sim RL 进行中（v9.8运行中，URDF夹爪bug已修复，稳定训练）**

---

## 一、项目全景

### 核心路线（经论文验证）

```
Phase 1: Isaac Lab sim RL（状态策略）
    ↓ 仿真成功率 > 80%
Phase 2: 真实机器人部署（状态策略 + 深度相机估位）
    ↓ 测试 sim-to-real gap
Phase 3: SmolVLA 微调（可选，需要泛化时）
    ↓ 25个真实演示微调
Phase 4: 边缘部署（地瓜S100P / Jetson Orin）
```

### 关键结论（本次讨论确认）

| 问题 | 结论 |
|------|------|
| 需要先做位姿估计吗？ | **不需要**。VLA端到端，RL策略只需物体xyz质心 |
| 一定要GR00T吗？ | **不需要**。SmolVLA(450M)更合适，8GB显卡可微调 |
| 需要收集大量真实演示吗？ | **不需要**。SmolVLA只需25个演示；纯RL可能不需要任何演示 |
| 边缘部署需要大GPU吗？ | **不需要**。我们的MLP策略CPU就能跑；SmolVLA在S100P上可量化部署 |
| Contact Sensor是所有夹爪都有的吗？| 仿真中任何机器人都可以加；真实硬件我们没有，用特权观测解决 |
| 有论文支撑这条路线吗？ | **有**，RLinf-Co/TwinRL-VLA/SimpleVLA-RL直接支持 |

---

## 二、训练版本历史与教训

### v8.x 局部最优级联

```
v8.4  → 侧向抓取局部最优（夹爪从侧面压物体）
v8.5  → 震荡（reaching_fine 与 grasped_ee_height 冲突）
v8.6  → 悬停+闭爪局部最优（grasp+height=120pts，lifting=0.87pts）
v8.7  → 完全失败（reaching_bonus_open权重>grasp权重，永不闭爪）
v8.8  → 夹爪始终闭合（URDF prismatic lower=0，Isaac Lab初始化为0）
v8.9  → 夹爪初始开启修复，但drop_penalty和grasp_threshold两个bug
v8.10 → 两个bug修复完成，训练5000iter完成
       最终结果：noise_std=1.36（未收敛），lifting=0.085（几乎没抬起）
       根本问题：5000iter远不够 + 无真实接触检测
```

### v9.x 修复级联（2026-03-05 ~ 2026-03-06）

| 版本 | 实验名 | 关键改动 | 结果 |
|------|--------|---------|------|
| v9.0 | arm_pick_place_v9_0 | 官方Isaac Lab 6-term奖励，18000iter | lifting=0（body-link夹取） |
| v9.1 | arm_pick_place_v9_1 | grasped_*奖励（夹爪gate），reach_penalty替换reach_bonus | 改善，但noise_std爆炸 |
| v9.2 | arm_pick_place_v9_2 | reaching_bonus_open（近物体+开爪=奖励） | 永不闭爪（reach_open>grasp权重） |
| v9.3 | arm_pick_place_v9_3 | 回退reaching_bonus，grasped_upvel防"抓不动"，entropy↓0.001 | noise_std=0.17，但随机初始化不稳 |
| v9.4 | arm_pick_place_v9_4 | EE offset 0.05→0.09（finger mid-grip真实距离） | grasp=0.37，object_height≈0 |
| v9.5 | arm_pick_place_v9_5 | resume=True从model_1000避免随机性 | 偶尔grasp=3.8，仍无法提升 |
| v9.6 | arm_pick_place_v9_5 | 诊断出URDF夹爪axis bug（右指与左指方向相同→间距永远3.2cm） | — |
| **v9.7** | arm_pick_place_v9_7 | **修复gripper_right_joint axis "-1 0 0"→"1 0 0"** | grasp 0→2.33，dropping 94%→10%（100轮内），然后LR=1e-3导致崩溃 |
| **v9.8** | arm_pick_place_v9_8 | **LR 1e-3→3e-4，clip_param 0.2→0.1，max_grad_norm 1.0→0.5** | 当前运行中，value_loss=0.10（健康） |

### 确认的核心Bug与修复

| Bug | 现象 | 修复 |
|-----|------|------|
| 夹爪初始闭合 | 机器人从不开爪 | init_state gripper_pos=0.04 (v8.9) |
| drop_penalty永不触发 | penalty(0.50) < termination(0.60) | penalty→0.72, termination→0.65 (v8.10) |
| grasp_threshold太大 | 0.12m时手指物理上够不到4cm宽物体 | threshold→0.09m (v8.10) |
| 抓取用距离判断(假抓) | 悬停靠近就触发所有奖励 | 需ContactSensor真实接触 (v9.0) |
| 训练步数不足 | 5000iter≈1亿步，操作任务需5-20亿步 | 18000iter (v9.0) |
| **EE offset偏短** | **EE帧在link_6+5cm，实际finger mid-grip~9cm，机器人被引导到错误位置** | **offset 0.05→0.09 (v9.4)** |
| **URDF夹爪axis bug** | **gripper_right_joint axis="-1 0 0"（与left相同）→两指同向移动，间距永远3.2cm，无法夹住4cm物体** | **axis→"1 0 0" (v9.7)，备份arm.urdf.bak_v9_6** |
| **学习率过高导致崩溃** | **LR=1e-3下，policy在grasp=2.33后振荡发散，value_loss飙251，action_rate=-8.3** | **LR→3e-4，clip_param→0.1，max_grad_norm→0.5 (v9.8)** |
| reaching_bonus_open冲突 | 近物体+开爪权重10>grasp权重5，永不闭爪 | 回退到标准reaching_bonus (v9.3) |
| noise_std爆炸 | entropy_coef=0.005推动noise_std→7.79+ | entropy_coef→0.001，init_noise_std→0.5 (v9.3) |

### 夹爪几何分析（v9.7修复依据）

```
gripper_left_joint:  axis="-1 0 0"，origin x=-0.0016279，left link CoM +1.56cm X（向右延伸）
gripper_right_joint: axis="+1 0 0"（已修复），origin x=-0.0016279，right link CoM -1.57cm X（向左延伸）

关闭(q=0): left CoM at +0.014, right CoM at -0.018 → gap=3.2cm < 物体4cm ✓
打开(q=0.04): left CoM at -0.026, right CoM at +0.022 → gap=4.8cm > 物体4cm ✓
```

---

## 三、Phase 1：仿真 RL（当前阶段）

### v9.0（下一步，立即部署）

**两个核心改动：接触感知 + 18000 iter**

#### 1. 接触传感器（仿真专用，不影响真实部署）

```python
# arm_cfg.py 修改
activate_contact_sensors=True   # False → True（一行改动）

# pick_place_cfg.py 添加 ContactSensorCfg
from isaaclab.sensors import ContactSensorCfg
gripper_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    history_length=3,
    update_period=0.0,
    track_air_time=True,
)
```

#### 2. 特权观测（Asymmetric Actor-Critic）

训练时Critic用接触力，部署时Actor不用，解决sim-to-real gap：

```
Critic obs（训练）: 普通obs(33维) + 接触力(6维) + 接触bool(1维) = 40维
Actor  obs（部署）: 仅普通obs(33维)
```

来源：Long-Horizon Privileged Action (2502.15442)

#### 3. 基于真实接触的Reward

```python
# 废弃：grasp_reward = (EE距物体 < 0.09m)  ← 假抓
# 新增：contact_grasp_reward
contact_force = contact_sensor.data.net_forces_w  # [N_envs, 3]
is_contacting = contact_force.norm(dim=-1) > 0.5  # 真实接触
is_closed     = gripper_pos < 0.01                 # 夹爪闭合
true_grasp    = is_contacting & is_closed           # 真正抓住
```

#### 4. 阶段课程（来源：Long-Horizon Privileged Action）

```
Stage 1 (iter 0-4000):    reach → 接触物体
  奖励：reaching_fine + contact_reward
  进阶：contact_force > 0.5N 成功率 > 70%

Stage 2 (iter 4000-9000): 接触 → 真实夹取
  奖励：+true_grasp_reward
  进阶：true_grasp 成功率 > 70%

Stage 3 (iter 9000-15000): 夹取 → 抬起 > 10cm
  奖励：+lifting_reward（真实接触gating）
  进阶：obj_z > 0.85m 成功率 > 70%

Stage 4 (iter 15000-18000): 抬起 → 放置到目标
  奖励：+goal_tracking_reward
```

#### 5. 训练配置

```python
# rsl_rl_ppo_cfg.py
num_iterations = 18000     # 5000 → 18000
entropy_coeff  = 0.005     # 0.01 → 0.005（减少随机探索）
```

---

### v9.1（v9.0完成后）：域随机化

目的：让仿真策略能够适应真实环境的物理差异

```python
# 物体物理随机化
mass:             [0.05, 0.3] kg    # 固定0.1kg → 随机
static_friction:  [0.8, 2.0]       # 固定1.5 → 随机
object_size:      ±20%             # 固定4x4x6cm → 随机

# 位置随机化（扩大）
pos_x: (-0.15, 0.15)               # (-0.08, 0.08) → 更大范围
pos_y: (-0.15, 0.15)               # (-0.10, 0.10) → 更大范围

# 关节噪声
joint_pos_noise: ±0.02 rad
joint_vel_noise: ±0.05 rad/s
```

---

## 四、Phase 2：真实机器人部署

**核心思路：直接部署状态RL策略，不需要视觉，不需要演示**

### 物体位置获取（不需要位姿估计）

```
深度相机（RealSense D435i）
    ↓
SAM2 分割 → 物体mask
    ↓
深度图提取mask区域深度 → 物体3D质心(x, y, z)
    ↓
输入到RL策略（替换仿真中的ground-truth位置）
```

只需要xyz质心，**不需要6D位姿**，抓取任务已经够用。

### Sim-to-Real Gap 处理

我们的RL策略（纯状态，无视觉）：
- 输入：关节位置 + 关节速度 + 物体xyz + 目标xyz + 上一动作
- 输出：关节位置增量 + 夹爪开/闭
- **视觉gap不存在**（不依赖图像）
- 主要gap：物体质量/摩擦力差异 → v9.1域随机化解决

### 硬件通信

```
机器人控制：Robstride motors → CAN bus → ControlCAN.dll
物体感知：RealSense D435i → USB3 → SAM2 → xyz质心
策略推理：MLP策略网络（<1ms/次） → 关节目标位置
```

---

## 五、Phase 3：SmolVLA 微调（可选）

**什么时候需要这步：** 当纯RL策略在真实环境不够泛化时（不同物体/位置/场景）

### 为什么选 SmolVLA 而不是 GR00T

| | SmolVLA | GR00T N1.6 |
|-|---------|-----------|
| 参数量 | 450M | 7B（16倍） |
| 微调显存 | **8GB** | 需要多卡A100 |
| 演示数量 | **25个** | 数百-数千 |
| 推理硬件 | CPU / 4GB GPU | A100级别 |
| LIBERO成绩 | 87.3% | 相当 |
| 开源 | ✅ HuggingFace Lerobot | ✅ NVIDIA |

### 微调流程

```bash
# 环境：HuggingFace Lerobot框架
# 硬件：本地RTX 5060(8GB) 或 SeetaCloud RTX 5090(32GB)

# 1. 在真实机器人上收集演示（25个够了）
# 数据格式：(RGB图, 深度图, 关节状态, 动作)

# 2. 微调SmolVLA
python lerobot/scripts/train.py \
    policy=smolvla \
    dataset_repo_id=your/dataset \
    --batch_size=16 \           # 8GB显卡用小batch
    --policy.use_amp=true        # 混合精度节省显存

# 3. 部署到S100P（量化）
# PyTorch → ONNX → 地瓜OpenExplorer(天工开物)工具链 → BPU
```

---

## 六、Phase 4：边缘部署

### 部署硬件选择

| 平台 | AI算力 | 内存 | 适合场景 | 备注 |
|------|--------|------|---------|------|
| **地瓜 RDK S100P** | 128 TOPS BPU | 24GB | 国产机器人，大小脑一体 | 2499元，MCU+BPU集成 |
| Jetson AGX Orin 64GB | 275 TOPS | 64GB | 国际主流，生态最好 | 最成熟 |
| Jetson Orin Nano 8GB | 40 TOPS | 8GB | 轻量任务 | 最便宜 |
| Jetson Thor | 1000 TOPS | 128GB | 人形机器人 | 旗舰 |

### 各策略的边缘部署可行性

| 策略 | 参数量 | S100P BPU | 说明 |
|------|--------|-----------|------|
| **我们的RL MLP** | ~1M | ✅ CPU都够 | 推理<1ms，完全无压力 |
| ACT | ~80M | ✅ 量化后可行 | Jetson Orin已有实测 |
| SmolVLA | 450M | ✅ 量化后可行 | 需PyTorch→ONNX→BPU转换 |
| GR00T N1.6 | 7B | ❌ 太大 | 需A100 |

### S100P部署流程（SmolVLA量化）

```
训练好的SmolVLA权重（PyTorch .pt）
    ↓ torch.onnx.export()
ONNX模型
    ↓ 地瓜 OpenExplorer（天工开物）工具链
BPU定点模型（INT8，关键算子FP16 VPU辅助）
    ↓
S100P BPU推理（128 TOPS，实测BPU占用率仅2%）
```

---

## 七、论文资料库

### 已下载论文（16篇，路径：`ref/papers/`）

**Reward设计 / 解决局部最优**
- `DrS_dense_reward_multistage_ICLR2024.pdf` — 判别器学reward，多阶段可复用
- `just_add_force_contact_rich_policies_2024.pdf` — 接触力作为reward信号
- `intrinsic_reward_sparse_reward_env_2026.pdf` — 稀疏奖励下的内在奖励
- `multistage_manipulation_demo_augmented_reward_2025.pdf` — 演示增强多阶段奖励

**Pick & Place / Sim-to-Real**
- `sim2real_pick_place_long_horizon_ICRA2025.pdf` — ICRA2025冠军（注：传统视觉+经典控制，非RL）
- `sim2real_dexterous_manipulation_humanoids_2025.pdf` — 视觉灵巧操作完整配方
- `long_horizon_manipulation_privileged_action_2025.pdf` — 特权动作+三阶段课程
- `real2sim2real_VLM_keypoint_reward_2025.pdf` — VLM自动生成keypoint reward

**VLA + RL（Phase 3/4参考）**
- `TwinRL_VLA_digital_twin_RL_2026.pdf` — 数字孪生驱动RL，20分钟100%成功
- `pi0_RL_online_finetuning_flow_VLA_2025.pdf` — π₀-RL，pick-place 41%→85%
- `VLA_RL_general_robotic_manipulation_2025.pdf` — VLA-RL端到端框架
- `SimpleVLA_RL_scaling_VLA_training_2025.pdf` — 纯sim数据训练VLA，每任务1个演示
- `GR_RL_dexterous_long_horizon_2024.pdf` — VLA专家化多阶段RL
- `RL100_real_world_RL_diffusion_manipulation_2025.pdf` — diffusion+PPO统一IL+RL

**框架/工程**
- `isaac_lab_GPU_sim_framework_2025.pdf` — Isaac Lab框架论文
- `RISE_self_improving_robot_policy_world_model_2026.pdf` — 自改进+组合世界模型

### 关键论文→改进映射

| 论文 | arxiv | 用在哪 | 核心insight |
|------|-------|--------|------------|
| Long-Horizon Privileged | 2502.15442 | v9.0 | 特权观测+三阶段课程 |
| DrS | 2404.16779 | v9.2 | 判别器自动学reward |
| Just Add Force | 2410.13124 | Phase 2真实 | 电机电流→接触力代理 |
| TwinRL-VLA | 2602.09023 | Phase 3 | sim RL→数字孪生→VLA |
| RLinf-Co | 2602.12628 | Phase 3 | sim RL+少量真实演示联合训练 |
| π₀-RL | 2510.25889 | Phase 4 | flow VLA在线RL微调 |
| SmolVLA | 2506.01844 | Phase 3 | 450M，8GB可微调，25演示够用 |
| π*₀.₆ | 2511.14759 | Phase 4 | RECAP：VLA从真实经验RL自我改进 |

---

## 八、PI系列VLA演进（Physical Intelligence）

| 版本 | 时间 | 方法 | 特点 |
|------|------|------|------|
| π₀ | 2024.10 | Flow Matching VLA | 高频50Hz，灵巧操作 |
| π₀-small | 2024.10 | 无VLM初始化版本 | 470M参数，轻量 |
| π₀.5 | 2025 | 异构数据联合训练 | 泛化到未见过的新环境 |
| **π*₀.₆** | **2025.11** | **RECAP（离线RL预训练）** | **从真实经验自我改进，最新** |
| openpi | 开源 | π₀/π₀.5权重开源 | github.com/Physical-Intelligence/openpi |

---

## 九、当前待办（优先级排序）

```
P0（当前）:  监控 v9.8 训练
             - 目标：lifting_object > 0（开始抬起）
             - 目标：reward > 100（进入goal_tracking阶段）
             - 约500轮后录视频确认行为
             - 实验目录：arm_pick_place_v9_8，max_iter=18000，LR=3e-4

P1（v9.8稳定后）: 域随机化（v9.9）
             - 物体质量/摩擦/大小随机化
             - 位置范围扩大

P2（仿真成功率>80%）: 真实机器人硬件调试
             - Robstride电机CAN通信
             - RealSense安装标定
             - SAM2物体质心估计

P3（真实部署后）: 评估是否需要SmolVLA
             - 如果泛化够用 → 结束
             - 如果需要泛化 → 收集25个演示，微调SmolVLA

P4（远期）:  π*₀.₆ RECAP方式在线RL自我改进
```

### 关键脚本说明

| 脚本 | 用途 |
|------|------|
| `play_record.py` | 录制训练策略视频（TiledCamera，headless） |
| `ssh_check.py` | 快速查看服务器训练进度 |
| `start_v98.py` | 启动v9.8训练（含checkpoint复制+PPO配置更新） |
| `server_code/arm_grasp/` | 服务器代码本地副本（与服务器同步） |

---

## 十、硬件资源总览

| 硬件 | 用途 | 备注 |
|------|------|------|
| SeetaCloud RTX 5090 32GB | Isaac Lab RL训练 | 约4万/月，按需租用 |
| 本地 RTX 5060 8GB | SmolVLA微调 | 8GB刚好够，用AMP+小batch |
| 地瓜 RDK S100P | 边缘推理部署 | 128 TOPS BPU，2499元 |
| RealSense D435i | 物体深度感知 | 已有/计划采购 |
