import gymnasium as gym

import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = DQN(stable_baselines3.dqn.MlpPolicy, vec_env, verbose=1)
model.learn(total_timesteps=500000)
model.save("../models/dqn_cartpole")

# model = DQN.load("../model/dqn_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, inf = vec_env.step(action)
    vec_env.render("human")