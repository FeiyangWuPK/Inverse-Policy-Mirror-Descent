from apmd_on.apmd import APMD
import numpy as np

import gym
import torch as th
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3 import SAC

env = gym.make('CartPole-v1')
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[32, 32], vf=[32, 32])])
model = APMD("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # env.render()
    if done:
      obs = env.reset()