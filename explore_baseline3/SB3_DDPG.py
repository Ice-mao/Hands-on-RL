import gymnasium as gym
import numpy as np

import stable_baselines3
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env

# Parallel Envs
vec_env = make_vec_env("Pendulum-v1")

# The noise objects for DDPG
n_actions = vec_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG(stable_baselines3.td3.policies.TD3Policy, vec_env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save("../models/ddpg_pendulum")

del model # remove to demonstrate saving and loading

model = DDPG.load("../models/ddpg_pendulum")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")