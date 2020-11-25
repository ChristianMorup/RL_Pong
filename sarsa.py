# SARSA
import sys
import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random

def epsilon_greedy(Q, state, nA, eps):
    if random.random() > eps:
        return np.argmax(Q[state])
    else:
        return random.choice(np.arange(nA))


def update_Q_sarsa(alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
    # Estimate in Q-table (for current state, action pair) Q(S_t,A_t)
    current = Q[state][action]
    # Get value of state, action pair at next time step Q(S_{t+1},A_{t+1})
    Qsa_next = Q[next_state][next_action] if next_state is not None else 0
    # Construct TD target R_{t+1} + gamma * Q(S_{t+1},A_{t+1})
    target = reward + (gamma * Qsa_next)
    # Get updated value Q(S_t,A_t) + alpha * (R_{t+1} + gamma * Q(S_{t+1},A_{t+1}) - Q(S_t,A_t))
    new_value = current + alpha * (target - current)

    return new_value