import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import highway_env
from matplotlib import pyplot as plt
import os
import numpy as np

# Create tensorboard log directory
tensorboard_log = "logs/"
os.makedirs(tensorboard_log, exist_ok=True)

def make_env():
    """Create a single environment instance"""
    def _init():
        env = gym.make("parallel-parking-v0", render_mode="rgb_array")
        return env
    return _init

def train_sac():
    """Train SAC with 8 parallel environments"""
    # Create 8 parallel environments
    n_envs = 8
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    
    print(f"Created {n_envs} parallel environments for training")
    
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50000,  # Increased buffer size for parallel training
        learning_starts=10000,
        batch_size=128,
        tau=0.005,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], qf=[128, 128]),
        ),
        ent_coef='auto',
        target_update_interval=1,
        target_entropy='auto',
        use_sde=False,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device='auto',
    )
    
    # Train for 500k steps across all environments
    model.learn(total_timesteps=500_000, log_interval=4, progress_bar=True)
    model.save("sac_parallel_parking_model")
    
    # Clean up
    env.close()
    return model

def evaluate_sac():
    """Evaluate the trained model"""
    # Create single environment for evaluation
    env = gym.make("parallel-parking-v0", render_mode="rgb_array")
    model = SAC.load("sac_parallel_parking_model")
    
    print("Starting evaluation...")
    
    episodes = 0
    max_episodes = 10  # Limit evaluation episodes
    
    obs, info = env.reset()
    
    while episodes < max_episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render if needed (optional - can be slow)
        # env.render()
        
        if terminated or truncated:
            episodes += 1
            print(f"Episode {episodes} completed")
            if episodes < max_episodes:
                obs, info = env.reset()
    
    env.close()
    print(f"Evaluation completed after {episodes} episodes")

def plot_training_progress():
    """Plot training progress from tensorboard logs"""
    # This function would read tensorboard logs and create plots
    # Implementation depends on your specific logging needs
    print("Training logs saved to:", tensorboard_log)
    print("Use 'tensorboard --logdir sac_parking_log' to view training progress")

if __name__ == "__main__":
    print("Starting SAC training with 8 parallel environments...")
    print("Training for 500,000 total timesteps...")
    
    # Train the model
    model = train_sac()
    
    print("Training completed! Starting evaluation...")
    
    # Evaluate the model
    evaluate_sac()
    
    # Show tensorboard info
    plot_training_progress()
    
    print("Done! Model saved as 'sac_parallel_parking_model.zip'")