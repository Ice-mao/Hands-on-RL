import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel Envs
vec_env = make_vec_env("CartPole-v1", n_envs=4)

#PPO继承自BaseAlgorithm
model = PPO(stable_baselines3.ppo.MlpPolicy, vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("../models/ppo_cartpole")

# 训练结果展示
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")