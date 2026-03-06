# VLA微调入门教程：自定义机械臂 → SmolVLA部署

> 来源整理自：HuggingFace LeRobot官方文档、SmolVLA论文、实战博客(2025-2026)
> 适用：6轴机械臂 + 深度相机 + pick-and-place任务

---

## 0. 整体流程

```
收集演示数据（遥操作）
    ↓ 50~200条轨迹
微调SmolVLA（LoRA，4小时）
    ↓
推理部署（Jetson AGX Orin或PC GPU）
```

三步搞定。不需要从头训练，不需要百万数据。

---

## 1. 理解SmolVLA

### 是什么
- HuggingFace出品，450M参数，开源
- 输入：RGB图像 + 语言指令 → 输出：关节动作序列
- 预训练在SO-100/SO-101等机械臂数据上（23k条轨迹）
- 比ACT、OpenVLA(7B)性能好，体积小10倍以上

### 硬件要求
| 阶段 | 显存需求 |
|------|---------|
| 微调训练 | 22GB（L4 GPU）/ 12GB（RTX 3080Ti，batch_size=16）|
| 推理部署 | 4~8GB（Jetson AGX Orin可跑）|

### 关键参数
- `n_action_steps = 50`：推理时每次预测50步动作（action chunking）
- `chunk_size = 30`：实际执行的chunk大小，太小（10-15）抓不住，太大（50）漂移
- 采样频率：10Hz（100ms/步）

---

## 2. 环境安装

```bash
# 推荐用uv（比pip快）
uv venv --python 3.10
source .venv/bin/activate

# 安装LeRobot（含SmolVLA支持）
uv pip install -e ".[feetech,smolvla]"

# 或者用pip
pip install lerobot[smolvla]
```

依赖：Python 3.10，CUDA 12.x，PyTorch 2.x

---

## 3. 遥操作采集数据

### 方式：主从臂（Leader-Follower）
最常用方式：用一个相同结构的"leader臂"手动控制，"follower臂"同步执行并录制。

```bash
# 先校准follower臂
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower

# 校准leader臂
lerobot-calibrate \
  --robot.type=so101_leader \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_leader

# 开始遥操作（测试）
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader
```

### 录制数据集
```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower \
  --robot.cameras="{
    front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30},
    wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}
  }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader \
  --display_data=true \
  --dataset.repo_id=your-hf-username/my-pickplace-dataset \
  --dataset.num_episodes=50 \
  --dataset.single_task="把杯子从高桌移到矮桌"
```

### 数据量建议
| 任务难度 | 建议数量 | 备注 |
|---------|---------|------|
| 简单pick-place | 50条 | 最低限，实测25条不够 |
| 正常任务 | 100~200条 | 推荐，泛化更好 |
| 复杂/多变场景 | 500条 | 物体位置变化大时 |

**关键技巧：**
- 每个物体位置录10条，共5个位置 = 50条（结构化采集效果更好）
- 每条轨迹大约20秒
- 录完后一定要**回放验证**（data replay），这步能发现大多数pipeline问题

### 自定义机械臂接入LeRobot
我们的机械臂不是SO-101，需要实现Robot接口：

```python
# 参考 lerobot/common/robot_devices/robots/robot.py
from lerobot.common.robot_devices.robots.robot import Robot

class MyCustomArm(Robot):
    def __init__(self, config):
        # 初始化CAN总线、电机等
        ...

    def connect(self):
        # 连接硬件
        ...

    def get_observation(self) -> dict:
        # 返回：joint_pos, joint_vel, 相机图像
        return {
            "observation.state": joint_positions,       # (8,) float
            "observation.images.front": front_image,    # (H,W,3) uint8
            "observation.images.wrist": wrist_image,    # (H,W,3) uint8
        }

    def send_action(self, action: dict):
        # 执行关节命令
        joint_cmds = action["action"]  # (8,) float
        # 发送到CAN总线...
```

参考：[LeRobot自定义硬件文档](https://huggingface.co/docs/lerobot/integrate_hardware)

---

## 4. 微调SmolVLA

### 上传数据集到HuggingFace
```bash
huggingface-cli login
# 数据集自动上传（lerobot-record会自动上传）
```

### 启动微调
```bash
cd lerobot
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=your-hf-username/my-pickplace-dataset \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla \
  --policy.device=cuda \
  --wandb.enable=true
```

**显存不够时：**
```bash
# 12GB显卡
--batch_size=16

# 8GB显卡（LoRA微调）
--policy.path=lerobot/smolvla_base \
--policy.use_lora=true \
--policy.lora_rank=16
```

### 训练时间参考
| GPU | batch_size | 时间 |
|-----|-----------|------|
| L4 (22GB) | 64 | ~4小时 |
| RTX 3080Ti (12GB) | 16 | ~8小时 |
| RTX 4090 (24GB) | 64 | ~3小时 |

### 修复config.json（重要）
训练完必须手动修改：
```bash
# 找到训练输出目录
vim outputs/train/my_smolvla/checkpoints/last/pretrained_model/config.json

# 修改这一行
"n_action_steps": 1  →  "n_action_steps": 50
```

### 上传模型
```bash
huggingface-cli upload your-hf-username/my-smolvla-pickplace \
  outputs/train/my_smolvla/checkpoints/last/pretrained_model pretrained_model
```

---

## 5. 推理部署

### 最重要的坑：用自己数据集的统计量
```python
# 错误做法：用smolvla_base的统计量
# 正确做法：用自己数据集的统计量

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("your-hf-username/my-pickplace-dataset")
stats = dataset.stats  # 用这个！
```

### 推理脚本
```python
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("your-hf-username/my-smolvla-pickplace")
policy.eval()
policy.to("cuda")

# 控制循环（10Hz）
while True:
    obs = robot.get_observation()

    # 每chunk_size步重新推理一次
    if step % chunk_size == 0:
        with torch.no_grad():
            action_chunk = policy.select_action(obs)

    action = action_chunk[step % chunk_size]
    robot.send_action(action)
    step += 1
    time.sleep(0.1)  # 10Hz
```

### Jetson AGX Orin部署
```bash
# Jetson上安装
pip install lerobot[smolvla]

# 推理（4-8GB显存，Orin可跑）
python inference.py \
  --policy=your-hf-username/my-smolvla-pickplace \
  --chunk_size=30
```

---

## 6. 调参建议

| 问题 | 原因 | 解决 |
|------|------|------|
| 抓不住 | chunk_size太小 | 从30开始，往上调 |
| 动作漂移 | chunk_size太大 | 降到20-25 |
| 成功率低 | 数据量不够 | 至少50条，推荐100条 |
| 推理抖动 | 统计量用错了 | 用自己数据集的stats |
| OOM | batch_size太大 | 减半，或开LoRA |

---

## 7. 我们项目的具体路线

```
当前阶段（Phase 1）：
  Isaac Lab仿真RL训练 → 机械臂学会抓取（正在进行）

Phase 2（下一步）：
  1. 实现LeRobot的Robot接口（CAN总线控制）
  2. 接RealSense深度相机，手眼标定
  3. 验证RL策略在真实机械臂可用
  4. 遥操作采集50~100条演示数据

Phase 3（微调VLA）：
  1. 用上面数据微调SmolVLA（用我们的5090，约4小时）
  2. 量化部署到Jetson AGX Orin（边缘推理）
  3. 支持自然语言指令："把杯子移到矮桌"
```

---

## 8. 关键资源

| 资源 | 链接 |
|------|------|
| SmolVLA官方文档 | https://huggingface.co/docs/lerobot/smolvla |
| LeRobot GitHub | https://github.com/huggingface/lerobot |
| 自定义硬件接入 | https://huggingface.co/docs/lerobot/integrate_hardware |
| SmolVLA微调实战(2026) | https://medium.com/correll-lab/fine-tuning-smolvla-for-new-environments-code-included-af266c56d632 |
| Pick-and-Place微调 | https://medium.com/@henryhu1607/genai-for-robotics-fine-tuning-smolvla-to-pick-and-place-940b485e6c9b |
| phospho完整教程 | https://docs.phospho.ai/learn/train-smolvla |
| LeRobot SO-101文档 | https://huggingface.co/docs/lerobot/so101 |
| SmolVLA预训练模型 | https://huggingface.co/lerobot/smolvla_base |
