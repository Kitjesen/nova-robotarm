"""Record video of trained pick-place policy (model_4500.pt)."""
import numpy as np
import torch
import imageio
import sys
import os

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser(description="Record pick-place policy video")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless", "--enable_cameras"])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

sys.path.insert(0, "/root/autodl-tmp/arm_grasp")
from arm_grasp.envs.pick_place_cfg import ArmPickPlaceEnvCfg_PLAY
from arm_grasp.agents.rsl_rl_ppo_cfg import ArmPickPlacePPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

MODEL_PATH = "/root/autodl-tmp/arm_grasp/logs/rsl_rl/arm_pick_place_v9_5/model_1000.pt"
VIDEO_PATH = "/root/autodl-tmp/arm_grasp/pick_place_v9_5_1000.mp4"
NUM_STEPS = 400
FPS = 30


@configclass
class RecordEnvCfg(ArmPickPlaceEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.scene.tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.7, -0.7, 1.1),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 20.0),
            ),
            width=800,
            height=600,
        )


def set_camera_lookat(env_path, eye, target):
    try:
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(f"{env_path}/Camera")
        if not cam_prim.IsValid():
            print(f"Camera not found at {env_path}/Camera")
            return
        xformable = UsdGeom.Xformable(cam_prim)
        xformable.ClearXformOpOrder()
        eye_v = Gf.Vec3d(*eye)
        target_v = Gf.Vec3d(*target)
        up = Gf.Vec3d(0, 0, 1)
        fwd = (target_v - eye_v).GetNormalized()
        right = (fwd ^ up).GetNormalized()
        new_up = (right ^ fwd).GetNormalized()
        mat = Gf.Matrix4d()
        mat.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
        mat.SetRow(1, Gf.Vec4d(new_up[0], new_up[1], new_up[2], 0))
        mat.SetRow(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0))
        mat.SetRow(3, Gf.Vec4d(eye_v[0], eye_v[1], eye_v[2], 1))
        xformable.AddTransformOp().Set(mat)
        print("Camera positioned.")
    except Exception as e:
        print(f"Camera warning: {e}")


def main():
    env_cfg = RecordEnvCfg()
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)

    set_camera_lookat("/World/envs/env_0", eye=(0.7, -0.7, 1.1), target=(0.15, 0.0, 0.85))

    # Load runner and get actor_critic directly
    runner_cfg = ArmPickPlacePPORunnerCfg()
    runner = OnPolicyRunner(env_wrapped, runner_cfg.to_dict(), log_dir=None, device="cuda")
    runner.load(MODEL_PATH)

    # Use policy directly for inference (act_inference expects full obs_dict)
    policy = runner.alg.policy
    policy.eval()
    print(f"Model loaded from {MODEL_PATH}")

    obs_dict, _ = env_wrapped.reset()
    frames = []
    camera = env.scene["tiled_camera"]

    for step in range(NUM_STEPS):
        with torch.no_grad():
            actions = policy.act_inference(obs_dict)
        obs_dict, _, dones, _ = env_wrapped.step(actions)

        # TiledCamera updates automatically with env.step; no extra render() needed
        img = camera.data.output.get("rgb", None)
        if img is not None and img.shape[0] > 0:
            frame = img[0].cpu().numpy()
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            frames.append(frame[:, :, :3].copy())

        if step % 50 == 0:
            print(f"Step {step}/{NUM_STEPS}, frames captured: {len(frames)}")

        if dones[0]:
            obs_dict, _ = env_wrapped.reset()

    env.close()

    if frames:
        imageio.mimwrite(VIDEO_PATH, frames, fps=FPS, quality=8)
        size_mb = os.path.getsize(VIDEO_PATH) / 1024 / 1024
        print(f"Video saved: {VIDEO_PATH} ({len(frames)} frames, {size_mb:.1f}MB)")
    else:
        print("No frames captured!")

    simulation_app.close()


if __name__ == "__main__":
    main()
