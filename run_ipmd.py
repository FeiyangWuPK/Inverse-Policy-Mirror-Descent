from utils.common import EvalCallback, linear_schedule, plot_costs
import os
import datetime

import gym
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from apmd_on.apmd import PMD
from apmd_on.iapmd import IPMD as IPMD_ON
from apmd_off.iapmd import IPMD as IPMD_OFF
from apmd_off.apmd import PMD as APMD_OFF

plt.style.use('ggplot')


def run_on_policy_iapmd(env_id='HalfCheetah-v4'):
    # on-policy
    expert_samples_replay_buffer_loc = f"logs/{env_id}-sac-buffer.pkl"

    env = make_vec_env(env_id, n_envs=1)
    # env = VecNormalize(env, norm_reward=False)
    ipmd_model = IPMD_ON("MlpPolicy", env, gamma=1.0, verbose=1, batch_size=256, train_freq=2048, learning_rate=linear_schedule(5e-3), gradient_steps=10, expert_replay_buffer_loc=expert_samples_replay_buffer_loc)

    eval_env = make_vec_env(env_id, n_envs=1)
    # eval_env = VecNormalize(eval_env, norm_reward=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path='logs/{}-apmd-on-ar/'.format(env_id),
                                 log_path='logs/{}-apmd-on-ar/'.format(env_id), eval_freq=1000,
                                 deterministic=True, render=False)

    ipmd_model.learn(total_timesteps=1e6, log_interval=10, )


# def run_off_policy_iapmd(env_id):
#     expert_samples_replay_buffer_loc = f"expert_trajectories/{env_id}/buffer5e6.pkl"
#
#     env = make_vec_env(env_id, n_envs=1)
#     ipmd_model = IPMD_OFF("MlpPolicy", env, gamma=1.0, verbose=1, batch_size=256, train_freq=1, learning_rate=linear_schedule(5e-3), gradient_steps=1, expert_replay_buffer_loc=expert_samples_replay_buffer_loc)
#
#     eval_env = make_vec_env(env_id, n_envs=1)
#     logtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     eval_callback = EvalCallback(eval_env, best_model_save_path=f'logs/iapmd/10traj/{env_id}-{logtime}/',
#                                  log_path=f'logs/iapmd/10traj/{env_id}-{logtime}/', eval_freq=500,
#                                  deterministic=True, render=False)
#     ipmd_model.learn(total_timesteps=2e6, log_interval=20, callback=eval_callback)

def run_off_policy_iapmd(env_id):
    expert_samples_replay_buffer_loc = f"expert_trajectories/{env_id}/buffer5e6.pkl"

    env = make_vec_env(env_id, n_envs=1)
    ipmd_model = IPMD_OFF("MlpPolicy", env, gamma=1.0, verbose=1, batch_size=512, train_freq=1, gradient_steps=5, expert_replay_buffer_loc=expert_samples_replay_buffer_loc)

    eval_env = make_vec_env(env_id, n_envs=1)
    logtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    eval_callback = EvalCallback(eval_env, best_model_save_path=f'logs/iapmd/10traj/{env_id}-{logtime}/',
                                 log_path=f'logs/iapmd/10traj/{env_id}-{logtime}/', eval_freq=500)

    ipmd_model.learn(total_timesteps=2e6, log_interval=20, callback=eval_callback)


def run_off_policy_apmd(env_id):
    env = make_vec_env(env_id, n_envs=1)
    ipmd_model = APMD_OFF("MlpPolicy", env, gamma=1.0, verbose=1)

    eval_env = make_vec_env(env_id, n_envs=1)
    logtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    eval_callback = EvalCallback(eval_env, best_model_save_path=f'logs/apmd/sac-style/{env_id}-{logtime}/',
                                 log_path=f'logs/apmd/sac-style/{env_id}-{logtime}/', eval_freq=500)
    ipmd_model.learn(total_timesteps=3e6, log_interval=5, callback=eval_callback)


if __name__ == "__main__":
    for env in ['Hopper-v4']:
        run_off_policy_apmd(env_id=env)
        # run_off_policy_iapmd(env_id=env)
