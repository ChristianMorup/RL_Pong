from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

apply_filters = True


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def create_title(t, m):
    if t == 'tracking':
        title = 'Reward: Tracking '
    elif t == 'hitting_the_ball':
        title = 'Reward: Hitting '
    else:
        title = 'Reward: Combination '
    if m == 'both':
        title += '(Both positive and negative reward)'
    elif m == 'negative':
        title += '(Only negative reward)'
    else:
        title += '(Only positive reward)'
    return 'Cumulative reward - ' + title, 'Score - ' + title


def create_title2(t):
    if t == 'tracking':
        title = 'Strategy: Tracking '
    elif t == 'hitting_the_ball':
        title = 'Strategy: Hitting '
    else:
        title = 'Strategy: Combination '
    return 'Cumulative reward - ' + title, 'Score - ' + title


def filter_by_picking_less_values(data):
    x = []
    y = []
    for i in range(len(data)):
        if i % 5 == 0:
            x = np.append(x, data[i])
            y = np.append(y, i)
    return x, y


reward_types = ['tracking', 'hitting_the_ball', 'combo']
modes = ['both', 'negative', 'positive']


def summarize_simulations(score_data, reward_data):
    score_sum = []
    reward_sum = []
    for column in range(1000):
        score_total = 0
        reward_total = 0
        for row in range(15):
            score_total += score_data[row][column]
            reward_total += reward_data[row][column]

        new_score_value = score_total // 15
        new_reward_value = reward_total // 15

        score_sum = np.append(score_sum, new_score_value)
        reward_sum = np.append(reward_sum, new_reward_value)

    if apply_filters:
        score_sum = moving_average(score_sum, 3)
        reward_sum = moving_average(reward_sum, 3)

        score_sum, score_episodes = filter_by_picking_less_values(score_sum)
        reward_sum, reward_episodes = filter_by_picking_less_values(reward_sum)
    else:
        score_episodes = range(1000)
        reward_episodes = range(1000)

    return score_sum, score_episodes, reward_sum, reward_episodes


for reward_type in reward_types:
    scores = []
    rewards = []
    episodes = []

    for mode in modes:
        score_file_name = 'sim_data/rev1/Simulation_' + reward_type + '_' + mode + '_score.csv'
        reward_file_name = 'sim_data/rev1/Simulation_' + reward_type + '_' + mode + '_reward.csv'

        s_data = genfromtxt(score_file_name, delimiter=';')
        r_data = genfromtxt(reward_file_name, delimiter=';')

        s_sum, episodes, r_sum, r_episodes = summarize_simulations(s_data, r_data)

        scores = np.append(scores, s_sum)
        rewards = np.append(rewards, r_sum)

    shape = (3, len(episodes))
    scores = scores.reshape(shape)
    rewards = rewards.reshape(shape)
    reward_title, score_title = create_title2(reward_type)  # create_title(reward_type, mode)


    plt.clf()
    plt.plot(episodes, scores[0], label="Both")
    plt.plot(episodes, scores[1], label="Negative")
    plt.plot(episodes, scores[2], label="Positive")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlim(xmin=0, xmax=1000)
    plt.ylim(ymin=0)
    plt.ylabel('Number of \'back wall\'-hits')
    plt.xlabel('Episodes')
    plt.title(score_title)
    plt.show()

    plt.clf()
    plt.plot(episodes, rewards[0].flatten(), label="Both")
    plt.plot(episodes, rewards[1].flatten(), label="Negative")
    plt.plot(episodes, rewards[2].flatten(), label="Positive")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlim(xmin=0, xmax=1000)
    plt.ylabel('Cumulative reward')
    plt.xlabel('Episodes')
    plt.title(reward_title)
    plt.show()
