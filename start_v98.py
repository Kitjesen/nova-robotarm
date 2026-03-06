"""Start v9.8 training: lower LR to prevent collapse."""
import paramiko
import time

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.westd.seetacloud.com', port=14918, username='root', password='UvUnT2x1jsaa', timeout=15)

def run(cmd, timeout=15):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(), stderr.read().decode()

# Step 1: Setup v9.8 checkpoint directory, starting from v9.7/model_0.pt
print("=== Setup v9.8 dir ===")
run("mkdir -p /root/autodl-tmp/arm_grasp/logs/rsl_rl/arm_pick_place_v9_8/")
run("cp /root/autodl-tmp/arm_grasp/logs/rsl_rl/arm_pick_place_v9_7/model_0.pt /root/autodl-tmp/arm_grasp/logs/rsl_rl/arm_pick_place_v9_8/model_0.pt")
out, _ = run("ls -la /root/autodl-tmp/arm_grasp/logs/rsl_rl/arm_pick_place_v9_8/")
print(out)

# Step 2: Write updated PPO config (v9.8)
ppo_cfg = '''"""RSL-RL PPO configurations for arm tasks."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ArmLiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "arm_lift_cube"
    run_name = ""
    resume = False
    logger = "tensorboard"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.0e-4, schedule="fixed", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class ArmPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """v9.8: Lower LR to prevent collapse (1e-3 -> 3e-4).
    Fixed URDF gripper axis, resume from v9.7/model_0 (v9.5/model_500 weights).
    Tighter clip_param=0.1, max_grad_norm=0.5 for stability.
    """
    num_steps_per_env = 24
    max_iterations = 18000
    save_interval = 500
    experiment_name = "arm_pick_place_v9_8"
    run_name = ""
    resume = True   # resume from model_0.pt in v9_8 dir
    logger = "tensorboard"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.1,
        entropy_coef=0.001,
        num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.0e-4, schedule="fixed", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=0.5,
    )
'''

sftp = client.open_sftp()
with sftp.file('/root/autodl-tmp/arm_grasp/arm_grasp/agents/rsl_rl_ppo_cfg.py', 'w') as f:
    f.write(ppo_cfg)
sftp.close()
print("PPO config (v9.8) written.")

# Step 3: Launch training
print("\n=== Launching v9.8 training ===")
stdin, stdout, stderr = client.exec_command(
    'bash -c "cd /root/autodl-tmp/arm_grasp && '
    'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
    'nohup /root/autodl-tmp/conda_envs/thunder2/bin/python -u '
    'scripts/train_pick_place.py --num_envs 2048 --headless '
    '> /root/autodl-tmp/arm_grasp/train_pick_place.log 2>&1 & echo PID:$!"',
    timeout=10
)
try:
    pid_out = stdout.read().decode()
    print("PID:", pid_out.strip())
except:
    pass

time.sleep(8)

# Verify
out, _ = run("ps aux | grep train_pick_place | grep python | grep -v grep", timeout=5)
print("Process:", out.strip()[:200] if out.strip() else "NOT FOUND")

# Initial log
time.sleep(5)
out, _ = run("tail -5 /root/autodl-tmp/arm_grasp/train_pick_place.log", timeout=10)
print("\nLog tail:")
print(out)

client.close()
print("Done!")
