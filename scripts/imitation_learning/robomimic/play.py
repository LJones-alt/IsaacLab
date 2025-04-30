# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import imageio
# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from isaaclab_tasks.utils import parse_env_cfg


def rollout(policy, env, horizon, device, render=False, video_writer=None, video_skip=5, camera_names=None):
    policy.start_episode
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])
    # compute reward
    video_count = 0  # video frame counter
    total_reward = 0.

    for i in range(horizon):
        # Prepare observations
        obs = obs_dict["policy"]
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])
        traj["obs"].append(obs)

        # Compute actions
        actions = policy(obs)
        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        #print(f"############ACTIONS ARE :  {actions}")
        obs_dict, r, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]
        
        total_reward += r
        #success = env.is_success()["task"]

        # visualization
        if render:
            env.render(mode="human", camera_name=camera_names[0])
        if video_writer is not None:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        if terminated:
            return True, traj
        elif truncated:
            return False, traj

    return False, traj, total_reward


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Set termination conditions
    env_cfg.terminations.time_out = None

    # Disable recorder
    env_cfg.recorders = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Load policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

    video_path = "rollout.mp4"
    video_writer = imageio.get_writer(video_path, fps=20)
    # Run policy
    results = []
    rewards = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
       # print(f"[INFO] Caught changes")
        terminated,  reward = rollout(policy, env, args_cli.horizon, device, False, None, 5, camera_names=["agentview"] )
        results.append(terminated)
        rewards.append(reward)
        print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")
    print(f"Rewards : {rewards} \n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
