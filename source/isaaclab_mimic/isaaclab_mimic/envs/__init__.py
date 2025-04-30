# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .franka_stack_ik_rel_blueprint_mimic_env_cfg import FrankaCubeStackIKRelBlueprintMimicEnvCfg
from .franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv
from .franka_stack_ik_rel_mimic_env_cfg import FrankaCubeStackIKRelMimicEnvCfg
from .Vial_ik_rel_blueprint_mimic_env_cfg import VialIKRelBlueprintMimicEnvCfg
from .Vial_ik_rel_mimic_env import VialIKRelMimicEnv
from .Vial_ik_rel_mimic_env_cfg import VialIKRelMimicEnvCfg
from .beaker_blueprint_mimic_env_cfg import BeakerBlueprintMimicEnvCfg
from .beaker_mimic_env_cfg import BeakerMimicEnvCfg
from .beaker_mimic_env import BeakerMimicEnv
##
# Inverse Kinematics - Relative Pose Control
##

# gym.register(
#     id="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
#     entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
#     kwargs={
#         "env_cfg_entry_point": franka_stack_ik_rel_mimic_env_cfg.FrankaCubeStackIKRelMimicEnvCfg,
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Vial-Pick-Place-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs:VialIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": Vial_ik_rel_mimic_env_cfg.VialIKRelMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Beaker-Mimic-v0",
    entry_point="isaaclab_mimic.envs:BeakerMimicEnv",
    kwargs={
        "env_cfg_entry_point": beaker_mimic_env_cfg.BeakerMimicEnvCfg,
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
#     entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
#     kwargs={
#         "env_cfg_entry_point": franka_stack_ik_rel_blueprint_mimic_env_cfg.FrankaCubeStackIKRelBlueprintMimicEnvCfg,
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Vial-IK-Rel-Blueprint-Mimic-v0",
    entry_point="isaaclab_mimic.envs:VialIKRelBlueprintMimicEnvCfg",
    kwargs={
        "env_cfg_entry_point": Vial_ik_rel_blueprint_mimic_env_cfg.VialIKRelBlueprintMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Beaker-Blueprint-Mimic-v0",
    entry_point="isaaclab_mimic.envs:BeakerBlueprintMimicEnvCfg",
    kwargs={
        "env_cfg_entry_point": beaker_blueprint_mimic_env_cfg.BeakerBlueprintMimicEnvCfg,
    },
    disable_env_checker=True,
)