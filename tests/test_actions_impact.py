import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from supply_chain.env import MFSCGymEnv

def run_episode_with_action(action_array, description):
    print(f"\n--- Testing Policy: {description} ---")
    print(f"Action Vector: {action_array}")
    
    env = MFSCGymEnv(step_size_hours=168, year_basis="thesis")
    obs, info = env.reset(seed=42)
    
    total_reward = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        obs, reward, done, truncated, info = env.step(action_array)
        total_reward += reward
        
    print(f"Total Reward (20 years): {total_reward:,.0f}")
    # We can fetch total backorders and delivered from the underlying sim
    print(f"Total Delivered:         {env.sim.total_delivered:,.0f}")
    print(f"Total Backorders:        {env.sim.total_backorders:,.0f}")
    
    return total_reward

if __name__ == "__main__":
    print("="*60)
    print("  DIAGNOSTIC: Do actions actually impact the simulation?")
    print("="*60)
    
    # Action space mapping: multipliers = 1.25 + 0.75 * action
    # action = -1.0 -> 0.5x (Half Q, Half ROP)
    # action =  0.0 -> 1.25x (Slightly higher than normal)
    # action = +1.0 -> 2.0x (Double Q, Double ROP)
    
    run_episode_with_action(np.array([-1.0, -1.0, -1.0, -1.0]), "EXTREME STARVATION (Half Inventory)")
    run_episode_with_action(np.array([0.0, 0.0, 0.0, 0.0]), "BASELINE (1.25x Inventory)")
    run_episode_with_action(np.array([1.0, 1.0, 1.0, 1.0]), "EXTREME HOARDING (Double Inventory)")

