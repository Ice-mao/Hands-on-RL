from stable_baselines3 import A2C

model = A2C("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log="../logs/a2c_cartpole_tensorboard/")
model.learn(total_timesteps=10_000)