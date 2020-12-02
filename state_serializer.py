import json
import os.path
from os import path
from collections import defaultdict
import numpy as np

def serialize_state(state):
    new_ball_state_str = f"{state[0][0]},{state[0][1]}"
    old_ball_state_str = f"{state[1][0]},{state[1][1]}"
    paddle_state_str = str(state[2])

    state_str = f"{new_ball_state_str};{old_ball_state_str};{paddle_state_str}"
    return state_str

def deserialize_state(state_str):
    states = state_str.split(";")

    new_ball_state_strings = states[0].split(",")
    print(new_ball_state_strings)
    new_ball_state = ((new_ball_state_strings[0]), int(new_ball_state_strings[1]))

    old_ball_state_strings = states[1].split(",")
    old_ball_state = (int(old_ball_state_strings[0]), int(old_ball_state_strings[1]))

    paddle_state = int(states[2])

    state = (new_ball_state, old_ball_state, paddle_state)
    return state

def save_Q_value(Q):
    x = dict((serialize_state(key), val.tolist()) for key, val in Q.items())

    q_json = json.JSONEncoder().encode(x)
    with open("./q_value.json", "w") as f:
        f.write(q_json)

def load_Q_value():
    nA = 3  # Up, Down, Do-nothing
    q_default = defaultdict(lambda: np.zeros(nA))
    if not path.exists("./q_value.json"):
        return q_default

    with open("./q_value.json", "r") as f:
        q_json = f.read()
        Q_with_str_keys = json.JSONDecoder().decode(q_json)

        Q = dict((deserialize_state(key), np.array(val)) for key, val in Q_with_str_keys.items())
        Q.update(q_default)
        return Q 