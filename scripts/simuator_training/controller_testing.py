# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/arms.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different single-arm manipulators.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import SceneEntityCfg
##
# Pre-defined configs
##
# isort: off
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
# isort: on


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )
    # articulation

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    
 
# def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
#     """Defines the origins of the the scene."""
#     # create tensor based on number of environments
#     env_origins = torch.zeros(num_origins, 3)
#     # create a grid of origins
#     num_rows = np.floor(np.sqrt(num_origins))
#     num_cols = np.ceil(num_origins / num_rows)
#     xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
#     env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
#     env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
#     env_origins[:, 2] = 0.0
#     # return the origins
#     return env_origins.tolist()

# def design_scene() -> tuple[dict, list[list[float]]]:
#     """Designs the scene."""
#     # Ground-plane
#     cfg = sim_utils.GroundPlaneCfg()
#     cfg.func("/World/defaultGroundPlane", cfg)
#     # Lights
#     cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
#     cfg.func("/World/Light", cfg)

#     # Create separate groups called "Origin1", "Origin2", "Origin3"
#     # Each group will have a mount and a robot on top of it
#     origins = define_origins(num_origins=1, spacing=2.0)

#       # Origin 1 with Franka Panda
#     prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
#     # -- Table
#     cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
#     cfg.func("/World/Origin1/Table", cfg, translation=(0.55, 0.0, 1.05))
#     # -- Robot
#     franka_arm_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Origin1/Robot")
#     franka_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
#     franka_panda = Articulation(cfg=franka_arm_cfg)
#     # -- glass beaker 
#     #prim_utils.create_prim("World/Origin2", "Xform", translation=(0.0,0.0,0.0))
#     beaker_cfg = sim_utils.UsdFileCfg(usd_path="source/orbit_assets/glass_beaker.usd")
#     beaker_cfg.func("/World/Objects/Beaker", beaker_cfg, translation=(0.3, 0.0, 1.05))


#     # return the scene information
#     scene_entities = {
#         "franka_panda": franka_panda
#     }
#     return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, scene:InteractiveScene):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
     # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=sim.device)

    robot=scene["robot"]

     # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    ee_goals = [
        [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
        [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
        [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    current_goal_idx = 0
    ## hold the commands here
    ik_commands = torch.zeros(1, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    ## set the robot
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)


    print(f"[INFO] DOING PHYSICS NOW ")
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities
            
            # root state
            root_state = robot.data.default_root_state.clone()
            
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply random actions to the robots
        
        # generate random joint positions
        joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
        joint_pos_target = joint_pos_target.clamp_(
            robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
        )
        # apply action to the robot
        robot.set_joint_position_target(joint_pos_target)
        # write data to sim
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
       
        robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    #scene_entities, scene_origins = design_scene()
    #scene_origins = torch.tensor(scene_origins, device=sim.device)
    scene_cfg = TableTopSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
