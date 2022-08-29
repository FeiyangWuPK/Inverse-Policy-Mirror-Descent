import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
n_episodes = int(1e7/2048)

def smooth(scalars, weight: float):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def plot_costs_with_band(rewards, names, smoothing_window=10, n=3, fig_name="acrobot.png", stds=None):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1,1,figsize=(7, 5))
    for i in range(n):
        extend = np.concatenate([np.ones(smoothing_window)*rewards[i][0], rewards[i]])
        rewards_smoothed = pd.Series(extend).rolling(smoothing_window, min_periods=smoothing_window).mean().to_numpy()
        rewards_smoothed = rewards_smoothed[smoothing_window:]
        # rewards_smoothed = rewards_smoothed[:1000]
        x = np.linspace(0, 1e7, num=n_episodes)
        if stds is None:
            ax.plot(x, rewards_smoothed, label=names[i], linewidth=3)
        else:
            std_smoothing_window = 1
            lower = rewards_smoothed - stds[i] * 0.95
            upper = rewards_smoothed + stds[i] * 0.95

            lower_extend = np.concatenate([np.ones(std_smoothing_window)*lower[0], lower])
            upper_extend = np.concatenate([np.ones(std_smoothing_window)*upper[0], upper])
            lower_smoothed = pd.Series(lower_extend).rolling(std_smoothing_window, min_periods=std_smoothing_window).mean().to_numpy()[std_smoothing_window:]
            upper_smoothed = pd.Series(upper_extend).rolling(std_smoothing_window, min_periods=std_smoothing_window).mean().to_numpy()[std_smoothing_window:]

            ax.plot(x, rewards_smoothed, label=names[i], linewidth=3)
            ax.fill_between(x, y1=lower_smoothed, y2=upper_smoothed, interpolate=False, alpha=0.5)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episodic Reward")
    plt.title(fig_name)
    plt.legend()
    plt.show()

env_names = ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2', 'Humanoid-v2',]
for env_name in env_names:
    if env_name == 'HalfCheetah-v2':
        ppo_mean = np.load('logs/HalfCheetah-v2-ppo-mean.npy')[:n_episodes]
        ppo_std = np.load('logs/HalfCheetah-v2-ppo-std.npy')[:n_episodes]
    else:
        ppo_result = np.load('logs/'+env_name+'-ppo/evaluations.npz')['results']
        ppo_mean = np.mean(ppo_result, axis=1)[:n_episodes]
        ppo_std = np.std(ppo_result, axis=1)[:n_episodes]

    trpo_result = np.load('logs/'+env_name+'-trpo/evaluations.npz')['results']
    trpo_mean = np.mean(trpo_result, axis=1)[:n_episodes]
    trpo_std = np.std(trpo_result, axis=1)[:n_episodes]
    mdpo_result = np.load('logs/'+env_name+'-mdpo/evaluations.npz')['results']
    mdpo_mean = np.mean(mdpo_result, axis=1)[:n_episodes]
    mdpo_std = np.std(mdpo_result, axis=1)[:n_episodes]
    plot_costs_with_band([ppo_mean, trpo_mean, mdpo_mean], names=['PPO', 'TRPO', 'MDPO'], n=3, smoothing_window=int(n_episodes/4), fig_name=env_name, stds=[ppo_std, trpo_std, mdpo_std])
