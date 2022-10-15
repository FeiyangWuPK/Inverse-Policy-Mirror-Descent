import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
import sys

if __name__ == '__main__':
    
    print(f'{sys.argv[1]}')
    env_id = sys.argv[1]
    env = make_vec_env(env_id, n_envs=1)
    sac_model = SAC("MlpPolicy", env, verbose=1)
    sac_model.learn(total_timesteps=5e6, log_interval=10)
    sac_model.save(f"logs/expert/{env_id}-sac/model")
    sac_model.save_replay_buffer(f"logs/expert/{env_id}-sac/buffer")