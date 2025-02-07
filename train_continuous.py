import os
import argparse
import h5py
import numpy as np
import torch

# robomimic imports
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

# stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# local imports
from env.latent_action_env import LatentActionEnv
from utils.losses import find_closest_value

def train_ppo(run_name, task_name, agent_path, rl_timesteps):
    """
    Main function to:
     1. Load Robomimic checkpoint + environment
     2. Load known embeddings (HDF5)
     3. Construct LatentActionEnv
     4. Train PPO on top of latent actions
    """

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    print(f"Running PPO for {run_name}, task={task_name}, device={device}")

    # (A) Load robomimic policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=agent_path,
        device=device,
        verbose=True
    )
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    horizon = config.experiment.rollout.horizon
    print(f"Horizon from config: {horizon}")

    # (B) Create environment from checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=None,
        render=False,
        render_offscreen=False,
        verbose=True
    )

    # (C) Load known embeddings
    embed_file = os.path.join(run_name, "known_embed_train.h5")
    if not os.path.isfile(embed_file):
        raise FileNotFoundError(f"Embedding file not found: {embed_file}")

    with h5py.File(embed_file, "r") as f:
        embeddings_array = np.array(f["embeddings"])
        values_array = np.array(f["values"])

    # (D) Create logs folder
    log_dir = os.path.join(run_name, "latent_ppo_logs")
    os.makedirs(log_dir, exist_ok=True)

    def make_env():
        return LatentActionEnv(
            env=env,
            policy=policy,
            horizon=horizon,
            embeddings_array=embeddings_array,
            values_array=values_array,
            log_dir=log_dir,
            task_name=task_name
        )

    vec_env = DummyVecEnv([make_env])

        # (F) Create and train PPO
    ppo_model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        n_steps=10,
        batch_size=128,
        learning_rate=1e-5,
        ent_coef=0.4,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5
    )
    print(f"Training PPO for {rl_timesteps} timesteps...")
    ppo_model.learn(total_timesteps=rl_timesteps)

    # (G) Save PPO model
    out_name = f"{run_name}/PPO_latent_{rl_timesteps}.zip"
    ppo_model.save(out_name)
    print(f"PPO model saved to: {out_name}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO with latent actions on a robomimic env.")
    parser.add_argument("--name", "-n", type=str, required=True,
                        help="Name of the run/folder (e.g. 'BC_can')")
    parser.add_argument("--task", "-t", type=str, required=True,
                        help="Task name (e.g. 'can', 'lift', etc.)")
    parser.add_argument("--agent", "-a", type=str, required=True,
                        help="Path to the robomimic agent checkpoint.")
    parser.add_argument("--rl_timesteps", "-r", type=int, default=3000,
                        help="Number of PPO training timesteps (default=3000)")
    args = parser.parse_args()

    train_ppo(
        run_name=args.name,
        task_name=args.task,
        agent_path=args.agent,
        rl_timesteps=args.rl_timesteps
    )


if __name__ == "__main__":
    main()
