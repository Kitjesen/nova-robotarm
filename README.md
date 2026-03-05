# NOVA Robot Arm

**inovxio** 自研 6-DOF 机械臂 —— Isaac Lab 强化学习训练 + 真机 gRPC 控制。

## 项目结构

```
robotarm_ws/
├── server_code/arm_grasp/      ← Isaac Lab RL 训练包
│   ├── envs/
│   │   ├── lift_cube_cfg.py    ← Task 1: 抬起方块
│   │   └── pick_place_cfg.py   ← Task 2: Pick & Place（当前 v8.10）
│   ├── mdp/rewards.py          ← 奖励函数
│   └── agents/rsl_rl_ppo_cfg.py← PPO 训练配置
├── robot_arm/
│   ├── grpc_stream/            ← 真机 gRPC 控制
│   │   ├── server.py           ← gRPC 服务端（接硬件）
│   │   ├── client.py           ← gRPC 客户端
│   │   ├── robstride.py        ← Robstride 电机驱动
│   │   └── Interpolation.py    ← 关节插值
│   └── kinematics/             ← 逆运动学求解
├── arm/                        ← ROS2 包（URDF/launch/meshes）
├── src/pkg_robotarm_py/        ← Python 控制包
├── start_training.py           ← 启动训练到 AutoDL 服务器
├── deploy_pick_place.py        ← 部署训练代码
└── arm_updated.urdf            ← 机械臂 URDF

```

## 快速开始

### 训练（Isaac Lab + RSL-RL）

```bash
# 部署并启动训练到 AutoDL 服务器
python start_training.py

# 本地仿真播放
python play_script.py
```

### 真机控制

```bash
# 启动 gRPC 服务端（在机械臂主控上运行）
python robot_arm/grpc_stream/server.py

# 客户端发送指令
python robot_arm/grpc_stream/client.py
```

## 训练任务进展

| 任务 | 环境 | 状态 |
|------|------|------|
| Lift Cube | `ArmLiftCubeEnvCfg` | ✅ 完成 |
| Pick & Place | `ArmPickPlaceEnvCfg` v8.10 | 🔵 训练中 |

## 硬件

- **电机**：Robstride 关节电机
- **控制**：gRPC（Protobuf）
- **运动学**：逆运动学求解器
- **仿真**：Isaac Lab（NVIDIA PhysX）

## 训练服务器

AutoDL RTX 5090：`connect.westd.seetacloud.com:14918`
路径：`/root/autodl-tmp/arm_grasp/`

## 相关项目

- [NOVA Dog](../nova-dog) — 机器狗整机（Fire Demo 中与机械臂协作）
- [AME-2](../../ame2_standalone) — 四足运动控制 RL
- [OTA 系统](../../infra/ota) — 固件远程升级
