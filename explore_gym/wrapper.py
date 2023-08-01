"""
try to study wrapper to fix the baseline3's connection with gym
"""

import gymnasium as gym
from gymnasium.wrappers import RescaleAction

base_env = gym.make("Hopper-v4")
print(base_env.action_space)
wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
print(wrapped_env.action_space)