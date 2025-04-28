# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG , FRANKA_PANDA_HIGH_PD_CFG # isort: skip
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg


@configclass
class VialSimplePickPlaceEnvCfg(LiftEnvCfg):
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.rack_pos = [0.3, 0.1, 0]
        self.vial_offset = [0.3,0,0.1] #[self.rack_pos[0]+0.002, self.rack_pos[1]+0.018, self.rack_pos[2]+0.018]
        self.rack_rot =[0.707, 0, 0, 0]

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (franka)
         # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        print("[DEBUG]  Robot actions set ...")
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"
        #self.commands.rack_pose.body_name = "panda_hand"

        # # Vial Rack - use when dynamic placement needed!
        # self.scene.rack = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Rack",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=self.rack_pos, rot=self.rack_rot),
        #     spawn=UsdFileCfg(
        #         usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/vial_rack.usd",
        #         scale=(1.0, 1.0, 1.0),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=24,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )

        # Use vial as object to pick
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=self.vial_offset, rot=self.rack_rot),
            spawn=UsdFileCfg(
                usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/test_cube.usd",
                #usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                semantic_tags=[("class", "object")],
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=24,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        print("[DEBUG]  Vial actions set ...")
       

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class VialSimplePickPlaceEnvCfg_PLAY(VialSimplePickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = True
