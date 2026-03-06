"""RSL-RL PPO configurations for arm tasks."""

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
    """v9.7: fix gripper_right_joint axis ("-1 0 0" -> "1 0 0") in URDF.
    Resume from v9.5 model_500 (reaching+grasp learned, now gripper actually grips).
    """
    num_steps_per_env = 24
    max_iterations = 18000
    save_interval = 500
    experiment_name = "arm_pick_place_v9_7"
    run_name = ""
    resume = True   # resume from model_0.pt (copied from v9_5/model_500)
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
        entropy_coef=0.001,
        num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="fixed", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )
