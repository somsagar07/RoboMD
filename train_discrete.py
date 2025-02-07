import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# robomimic imports
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

# Local import of our environment
from env.can_env import CanEnv
from env.lift_env import LiftEnv
from env.thread_env import ThreadEnv
from env.square_env import SquareEnv
from env.stack_env import StackEnv

TASK_ENVS = {
    "can": CanEnv,
    "lift": LiftEnv,
    "thread": ThreadEnv,
    "square": SquareEnv,
    "stack": StackEnv,
}


def main(args):
    # Prepare directories
    log_dir = f"{args.task_name}_logs"
    data_dir = f"{args.task_name}_rl_data"

    # Create required directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, args.save_path), exist_ok=True)

    if args.collect_data:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, args.save_path), exist_ok=True)

    # Print arguments for clarity
    print(f"Agent Path: {args.agent_path}")
    print(f"Video Record: {args.video_record}")
    print(f"RL Update Step: {args.rl_update_step}")
    print(f"RL Timesteps: {args.rl_timesteps}")
    print(f"Collect Data: {args.collect_data}")
    print(f"Render: {args.render}")

    # Load the policy checkpoint
    ckpt_path = args.agent_path
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # Determine horizon (if not specified, read from config)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    rollout_horizon = config.experiment.rollout.horizon

    # Create environment from the checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=None,
        render=args.render,
        render_offscreen=False,  # If you need offscreen rendering set to True
        verbose=True
    )

    # Instantiate our custom environment
    env_class = TASK_ENVS.get(args.task_name)
    if env_class is None:
        raise ValueError(f"Unknown task name: {args.task_name}")

    def make_rl_env():
        return env_class(
            env=env,
            policy=policy,
            rollout_horizon=rollout_horizon,
            video_record=args.video_record,
            collect_data=args.collect_data,
            save_path=args.save_path,
            device=device
        )

    # Create a VecEnv for stable-baselines
    vec_env = DummyVecEnv([make_rl_env])

    # Create the PPO model
    ppo_model = PPO("CnnPolicy", vec_env, verbose=1, n_steps=args.rl_update_step)
    ppo_model.learn(total_timesteps=args.rl_timesteps)

    # Save the trained model
    os.makedirs("trained_rl_models", exist_ok=True)
    save_model_path = f"trained_rl_models/{args.save_path}_ppo_model_{args.rl_timesteps}"
    ppo_model.save(save_model_path)
    print(f"Training completed. Model saved to {save_model_path}")

    # Example: get action log probabilities
    observation = vec_env.reset()
    observation = torch.tensor(observation).float().to(ppo_model.device)
    with torch.no_grad():
        dist = ppo_model.policy.get_distribution(observation)
    n_actions = vec_env.action_space.n
    all_actions = torch.arange(n_actions).to(ppo_model.device)
    log_probs = dist.log_prob(all_actions)
    print("Log Probabilities of All Actions:", log_probs)

    if args.save_logs:
        log_file = os.path.join(log_dir, args.save_path, "log_prob.txt")
        with open(log_file, "a") as file:
            file.write(str(log_probs) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent to find failure modes for robosuite task.")

    parser.add_argument("--task_name", type=str, default="can", help="Name of the task/environment. Used to form directory names (e.g., can_logs, can_rl_data).")
    parser.add_argument("--agent_path", type=str, required=True, help="Path to load the agent checkpoint (.pt file).")
    parser.add_argument("--rl_timesteps", type=int, default=300, help="Number of training timesteps (default: 300)")
    parser.add_argument("--rl_update_step", type=int, default=300, help="Number of steps per PPO update (default: 300)")
    parser.add_argument("--video_record", action="store_true", help="Record training video if set.")
    parser.add_argument("--render", action="store_true", help="Render rollout if set.")
    parser.add_argument("--collect_data", action="store_true", help="If set, collect image data during rollouts.")
    parser.add_argument("--save_path", type=str, default="default_run", help="Folder name to save logs and data.")
    parser.add_argument("--save_logs", action="store_true", help="Save logs.")

    args = parser.parse_args()
    main(args)
