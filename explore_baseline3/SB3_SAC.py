import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("Pendulum-v1", n_envs=4)
# env = gym.make("Pendulum-v1", render_mode="human")

model = SAC("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=500000, log_interval=4)
model.save("../models/sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("../models/sac_pendulum")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info  = vec_env.step(action)
    vec_env.render("human")