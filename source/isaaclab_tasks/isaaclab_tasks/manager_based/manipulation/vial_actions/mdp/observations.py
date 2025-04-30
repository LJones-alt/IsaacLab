# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    object: RigidObject = env.scene[object_cfg.name]

    return torch.cat((object.data.root_pos_w), dim=1)


def instance_randomize_object_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
    
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    object: RigidObjectCollection = env.scene[object_cfg.name]
    

    object_pos_w = []
    for env_id in range(env.num_envs):
        object_pos_w.append(object.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        
    object_pos_w = torch.stack(object_pos_w)
    

    return torch.cat((object_pos_w), dim=1)


def object_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
   
):
    """The orientation of the cubes in the world frame."""
    object: RigidObject = env.scene[object_cfg.name]
    

    return torch.cat((object.data.root_quat_w), dim=1)


def instance_randomize_object_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    
) -> torch.Tensor:
    """The orientation of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    object: RigidObjectCollection = env.scene[object_cfg.name]
   
    object_quat_w = []
    
    for env_id in range(env.num_envs):
        object_quat_w.append(object.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        
    object_quat_w = torch.stack(object_quat_w)


    return torch.cat((object_quat_w), dim=1)


def object_obs(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        object pos,
        object quat,
        c
        gripper to object,
        
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    object_quat_w = object.data.root_quat_w


    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_object = object_pos_w - ee_pos_w
    return torch.cat(
        (
            object_pos_w - env.scene.env_origins,
            object_quat_w,
            gripper_to_object,
        ),
        dim=1,
    )


def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        object pos,
        object quat,
        cube_2 pos,
        cube_2 quat,
        cube_3 pos,
        cube_3 quat,
        gripper to object,
        gripper to cube_2,
        gripper to cube_3,
        object to cube_2,
        cube_2 to cube_3,
        object to cube_3,
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    object: RigidObjectCollection = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = []
  
    object_quat_w = []
   
    for env_id in range(env.num_envs):
        object_pos_w.append(object.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        
        object_quat_w.append(object.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        
    object_pos_w = torch.stack(object_pos_w)
    
    object_quat_w = torch.stack(object_quat_w)
    
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_object = object_pos_w - ee_pos_w

    return torch.cat(
        (
            object_pos_w - env.scene.env_origins,
            object_quat_w,
            gripper_to_object
        ),
        dim=1,
    )


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
    )

    return grasped


def object_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
) -> torch.Tensor:
    """Check if an object is stacked by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
    )
    stacked = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
    )

    return stacked