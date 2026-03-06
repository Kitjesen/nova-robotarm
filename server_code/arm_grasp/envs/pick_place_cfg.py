"""Pick-and-Place v9.0: official Isaac Lab reward structure.

Strategy: mirror the proven 6-term Isaac Lab lift reward.
No grasp detection gating — robot learns to grasp implicitly
by being rewarded for lifting height (object_is_lifted fires
only when object actually moves up, requires physical contact).

Key changes from v8.10:
  - Removed: grasp, grasped_upvel, grasped_height, grasped_goal,
              drop_penalty, joint_deviation (11 terms → 6 terms)
  - object_is_lifted: binary reward weight=15 (official: 15)
  - reaching: std=0.1 weight=1 (official: 1)
  - object_goal_distance: std=0.3/0.05 (official: 0.3/0.05)
  - minimal_height=0.82 (object starts at z=0.78, table top=0.75)
  - max_iterations: 5000 → 18000
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from arm_grasp import mdp
from arm_grasp.assets import ARM_6DOF_CFG


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):

    robot = ARM_6DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.75)
    robot.init_state.joint_pos = {
        "joint_1": 0.0,
        "joint_2": 0.0,
        "joint_3": 0.0,
        "joint_4": 0.0,
        "joint_5": 0.0,
        "joint_6": 0.0,
        "gripper_left_joint": 0.04,
        "gripper_right_joint": 0.04,
    }

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link_6",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.09)),  # gripper finger mid-grip point (~9cm from link_6)
            ),
        ],
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, 0.78), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.06),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.27, 0.07)),
        ),
    )

    table_high = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableHigh",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.375)),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.75),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
        ),
    )

    table_low = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLow",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.275)),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.55),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.50, 0.55)),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_6",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.35), pos_y=(-0.15, 0.15), pos_z=(0.10, 0.30),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        scale=2.0,
        use_default_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_left_joint", "gripper_right_joint"],
        open_command_expr={"gripper_left_joint": 0.04, "gripper_right_joint": 0.04},
        close_command_expr={"gripper_left_joint": 0.0, "gripper_right_joint": 0.0},
    )


@configclass
class ObservationsCfg:
    """Asymmetric Actor-Critic observations.

    Actor (policy): observations available on real robot via FK + depth camera.
    Critic (privileged): adds object velocity/angular velocity — hard to measure
    on real hardware without IMU on object. Critic is discarded at deployment.

    Reference: Privileged Action paper (arXiv 2502.15442), DrS (ICLR 2024).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations — all available on real robot."""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)           # 8D
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)           # 8D
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)  # 3D
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_frame)               # 3D
        ee_to_object = ObsTerm(func=mdp.ee_to_object_vector)                     # 3D
        gripper_opening = ObsTerm(func=mdp.gripper_opening)                      # 2D
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )                                                      # 7D
        actions = ObsTerm(func=mdp.last_action)               # 7D
        # Total: 41D

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations — actor obs + privileged info (training only)."""
        # Same as actor
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_frame)
        ee_to_object = ObsTerm(func=mdp.ee_to_object_vector)
        gripper_opening = ObsTerm(func=mdp.gripper_opening)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)
        # Privileged: object dynamics (unknown on real robot without sensors)
        object_lin_vel = ObsTerm(func=mdp.object_velocity_in_robot_frame)  # 3D
        object_ang_vel = ObsTerm(func=mdp.object_ang_vel_w)                # 3D
        # Total: 47D

        def __post_init__(self):
            self.enable_corruption = False  # no noise for critic
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.10, 0.10), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """v9.1: lift_cube proven reward structure, adapted for pick_place scene.

    Key insight: reaching_penalty (-distance) has constant gradient at any
    distance. reaching_bonus (1-tanh) gradient vanishes >0.5m — unusable
    when arm starts far from object. lift_cube used reaching_penalty and
    succeeded (reward +34.3). Copying that structure here.
    """

    # Global approach: -distance, constant gradient regardless of how far
    reaching = RewTerm(func=mdp.reaching_penalty, weight=5.0)
    # Fine approach: standard proximity bonus (gripper state irrelevant here)
    reaching_fine = RewTerm(
        func=mdp.reaching_bonus,
        params={"std": 0.1},
        weight=10.0,
    )

    # Grasp: close gripper when EE near object
    grasp = RewTerm(
        func=mdp.grasp_reward,
        params={"threshold": 0.08, "open_pos": 0.04},
        weight=5.0,
    )

    # Lift incentive: reward upward velocity of object when grasped.
    # Breaks "grab and hold" — robot must move up to earn reward after grasping.
    grasped_upvel = RewTerm(
        func=mdp.grasped_object_upvel_reward,
        params={"scale": 0.3, "grasp_threshold": 0.08, "open_pos": 0.04},
        weight=10.0,
    )

    # Continuous height gain: gated on gripper actually holding the object
    object_height = RewTerm(
        func=mdp.grasped_height_reward,
        params={"initial_height": 0.78, "max_bonus_height": 1.0,
                "grasp_threshold": 0.08, "open_pos": 0.04},
        weight=10.0,
    )
    # Binary lift bonus: gated on gripper holding
    lifting_object = RewTerm(
        func=mdp.grasped_and_lifted,
        params={"minimal_height": 0.82, "grasp_threshold": 0.08, "open_pos": 0.04},
        weight=5.0,
    )

    # Goal tracking: gated on gripper holding
    object_goal_tracking = RewTerm(
        func=mdp.grasped_goal_distance,
        params={"std": 0.5, "minimal_height": 0.82, "grasp_threshold": 0.08,
                "open_pos": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine = RewTerm(
        func=mdp.grasped_goal_distance,
        params={"std": 0.05, "minimal_height": 0.82, "grasp_threshold": 0.08,
                "open_pos": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.65, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class ArmPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=2048, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class ArmPickPlaceEnvCfg_PLAY(ArmPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
