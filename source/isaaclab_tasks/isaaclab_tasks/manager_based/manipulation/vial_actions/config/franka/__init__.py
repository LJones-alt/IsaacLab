# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    agents,
    vial_ik_rel_blueprint_env_cfg,
    vial_ik_rel_env_cfg,
    vial_ik_rel_instance_randomize_env_cfg,
    vial_joint_pos_env_cfg,
    vial_joint_pos_instance_randomize_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Vial-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": vial_joint_pos_env_cfg.VialEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Vial-Instance-Randomize-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": vial_joint_pos_instance_randomize_env_cfg.VialInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Vial-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": vial_ik_rel_env_cfg.VialEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Vial-Randomize-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": vial_ik_rel_instance_randomize_env_cfg.VialInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Vial-Franka-IK-Rel-Blueprint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": vial_ik_rel_blueprint_env_cfg.VialBlueprintEnvCfg,
    },
    disable_env_checker=True,
)
