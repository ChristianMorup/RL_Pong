import numpy as np
import matplotlib.pyplot as plt

from file_util import FileUtil
from modes_and_types import *
apply_filters = True

#TODO:
## Proportional to the distance
## Font and width of figures is too small
## Put labels on top of the figure
## Number of missed balls
## Invert the scores

reward_types = all_rewards
modes = all_modes

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def map_reward_type_to_title(reward):
    if reward == RewardType.TRACKING:
        title = 'Simple tracking '
    elif reward == RewardType.HITTING:
        title = 'Hitting '
    elif reward == RewardType.TRACKING_AND_HITTING:
        title = 'Simple tracking + Hitting '
    elif reward == RewardType.TRACKING_PROPORTIONAL:
        title = 'Tracking \n(Proportional to the distance - Unidirectional) '
    elif reward == RewardType.TRACKING_PROPORTIONAL_UNIDIRECTIONAL:
        title = 'Tracking \n(Proportional to the distance - Both directions) '
    elif reward == RewardType.TRACKING_PROPORTIONAL_UNIDIRECTIONAL_WEIGHTED:
        title = 'Tracking \n(Weighted and proportional to the distance - Unidirectional)'
    elif reward == RewardType.TRACKING_STEADY_NERVES:
        title = 'Steady nerves'
    else:
        title = ''
    return 'Cumulative reward - ' + title, 'Score - ' + title


def filter_by_picking_less_values(data):
    x = []
    y = []
    for i in range(len(data)):
        if i % 5 == 0:
            x = np.append(x, data[i])
            y = np.append(y, i)
    return x, y


def summarize_simulations(score_data, reward_data, n_simulations=15):
    score_sum = []
    reward_sum = []

    # lowest_reward = find_lowest_first_reward_score(reward_data, n_simulations)
    for column in range(1000):
        score_total = 0
        reward_total = 0
        for row in range(n_simulations):
            score_total += (100 - score_data[row][column])
            reward_total += reward_data[row][column]
            #reward_total += normalize_reward_score(lowest_reward, reward_data[row][column])

        new_score_value = score_total // n_simulations
        new_reward_value = reward_total // n_simulations

        score_sum = np.append(score_sum, new_score_value)
        reward_sum = np.append(reward_sum, new_reward_value)

    return filter(score_sum, reward_sum)


def summarize_reward_simulations(reward_data, n_simulations=15):
    reward_sum = []

    # lowest_reward = find_lowest_first_reward_score(reward_data, n_simulations)
    for column in range(1000):
        reward_total = 0
        for row in range(n_simulations):
            reward_total += reward_data[row][column]
            # reward_total += normalize_reward_score(lowest_reward, reward_data[row][column])

        new_reward_value = reward_total // n_simulations

        reward_sum = np.append(reward_sum, new_reward_value)

    reward_sum = moving_average(reward_sum, 3)

    reward_sum, reward_episodes = filter_by_picking_less_values(reward_sum)

    return reward_sum, reward_episodes


def find_lowest_first_reward_score(data, n_simulations=15):
    lowest_reward = 0
    for row in range(n_simulations):
        if lowest_reward > data[row][0]:
            lowest_reward = data[row][0]
    return lowest_reward


def normalize_reward_score(lowest_reward, value):
    return value - lowest_reward


def filter(score_sum, reward_sum):
    if apply_filters:
        score_sum = moving_average(score_sum, 3)
        reward_sum = moving_average(reward_sum, 3)

        score_sum, score_episodes = filter_by_picking_less_values(score_sum)
        reward_sum, reward_episodes = filter_by_picking_less_values(reward_sum)
    else:
        score_episodes = range(1000)
        reward_episodes = range(1000)

    return score_sum, score_episodes, reward_sum, reward_episodes


def plot_simulations(n_simulations=10):
    for reward_type in reward_types:
        scores = []
        rewards = []
        pos_rewards = []
        neg_rewards = []
        episodes = []

        for mode in modes:
            file_util = FileUtil('sim_data/test/')
            s_data, r_data, r_pos_data, r_neg_data = file_util.read_files(reward_type, mode)

            s_sum, episodes, r_sum, r_episodes = summarize_simulations(s_data, r_data, n_simulations=n_simulations)
            scores = np.append(scores, s_sum)
            rewards = np.append(rewards, r_sum)

            if len(r_pos_data) > 0:
                r_pos_sum, pos_episodes = summarize_reward_simulations(r_pos_data, n_simulations=n_simulations)
                pos_rewards = np.append(pos_rewards, r_pos_sum)

            if len(r_neg_data) > 0:
                r_neg_sum, neg_episodes = summarize_reward_simulations(r_neg_data, n_simulations=n_simulations)
                neg_rewards = np.append(neg_rewards, r_neg_sum)

        shape = (3, len(episodes))
        scores = scores.reshape(shape)
        rewards = rewards.reshape(shape)

        if len(pos_rewards) > 0:
            shape = (2, len(episodes))
            pos_rewards = pos_rewards.reshape(shape)
        if len(neg_rewards) > 0:
            shape = (2, len(episodes))
            neg_rewards = neg_rewards.reshape(shape)

        reward_title, score_title = map_reward_type_to_title(reward_type, mode)  # create_title(reward_type, mode)

        plt.clf()
        # plt.plot(episodes, scores, label="Both")
        plt.plot(episodes, scores[0], label="Both")
        plt.plot(episodes, scores[1], label="Negative")
        plt.plot(episodes, scores[2], label="Positive")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True, shadow=False)
        plt.xlim(xmin=0, xmax=1000)
        plt.ylim(ymin=0, ymax=100)
        plt.ylabel('n Hits (max=100)')
        plt.xlabel('Episodes')
        plt.title(score_title)
        plt.show()

        plt.clf()
        # plt.plot(episodes, rewards, label="Both")
        plt.plot(episodes, rewards[0].flatten(), label="Both")
        plt.plot(episodes, rewards[1].flatten(), label="Negative")
        plt.plot(episodes, rewards[2].flatten(), label="Positive")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True, shadow=False)
        plt.xlim(xmin=0, xmax=1000)
        plt.ylabel('Cumulative reward')
        plt.xlabel('Episodes')
        plt.title(reward_title)
        plt.show()

        if len(pos_rewards) > 0:
            plt.clf()
            # plt.plot(episodes, rewards, label="Both")
            plt.plot(episodes, pos_rewards[0].flatten(), label="Both")
            plt.plot(episodes, pos_rewards[1].flatten(), label="Positive")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True, shadow=False)
            plt.xlim(xmin=0, xmax=1000)
            plt.ylabel('Cumulative positive reward')
            plt.xlabel('Episodes')
            plt.title(reward_title)
            plt.show()

        if len(neg_rewards) > 0:
            plt.clf()
            # plt.plot(episodes, rewards, label="Both")
            plt.plot(episodes, neg_rewards[0].flatten(), label="Both")
            plt.plot(episodes, neg_rewards[1].flatten(), label="Negative")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True, shadow=False)
            plt.xlim(xmin=0, xmax=1000)
            plt.ylabel('Cumulative negative reward')
            plt.xlabel('Episodes')
            plt.title(reward_title)
            plt.show()


def do_simple_plot(score):
    plt.clf()
    plt.plot(range(len(score)), score, label="0.1")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlim(xmin=0, xmax=400)
    plt.ylim(ymin=0)
    plt.ylabel('Number of \'back wall\'-hits')
    plt.xlabel('Episodes')
    plt.title('Test')
    plt.show()


plot_simulations()
