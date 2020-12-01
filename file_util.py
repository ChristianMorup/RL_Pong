import csv

from modes_and_types import *
from numpy import genfromtxt
import numpy as np

class FileUtil:
    folder_path = ''

    def __init__(self, folder_path=''):
        self.folder_path = folder_path

    def get_file_names(self, reward_type, mode):
        score = self.folder_path + 'Simulation_' + reward_type_map.get(
            reward_type) + '_' + mode_map.get(mode) + '_score.csv'
        reward = self.folder_path + 'Simulation_' + reward_type_map.get(
            reward_type) + '_' + mode_map.get(mode) + '_reward.csv'

        pos_reward = ''
        neg_reward = ''
        if mode == Mode.BOTH or mode == Mode.POSITIVE:
            pos_reward = self.folder_path + 'Simulation_' + reward_type_map.get(
                reward_type) + '_' + mode_map.get(mode) + '_pos_reward.csv'
        if mode == Mode.BOTH or mode == Mode.NEGATIVE:
            neg_reward = self.folder_path + 'Simulation_' + reward_type_map.get(
                reward_type) + '_' + mode_map.get(mode) + '_neg_reward.csv'

        return score, reward, pos_reward, neg_reward

    def write_to_csv(self, reward_type, mode, **kwargs):
        score_file_name, reward_file_name, pos_reward_file_name, neg_reward_file_name = self.get_file_names(reward_type,
                                                                                                            mode)

        f = open(score_file_name, 'a')
        csv.writer(f, delimiter=';').writerow(kwargs.get('cum_score'))
        f.close()

        f = open(reward_file_name, 'a')
        csv.writer(f, delimiter=';').writerow(kwargs.get('cum_reward'))
        f.close()

        if mode == Mode.BOTH or mode == Mode.POSITIVE:
            f = open(pos_reward_file_name, 'a')
            csv.writer(f, delimiter=';').writerow(kwargs.get('cum_pos_reward'))
            f.close()

        if mode == Mode.BOTH or mode == Mode.NEGATIVE:
            f = open(neg_reward_file_name, 'a')
            csv.writer(f, delimiter=';').writerow(kwargs.get('cum_neg_reward'))
            f.close()

    def read_files(self, reward_type, mode):
        score_file_name, reward_file_name, pos_reward_file_name, neg_reward_file_name = self.get_file_names(reward_type,
                                                                                                            mode)
        score_data = genfromtxt(score_file_name, delimiter=';')
        reward_data = genfromtxt(reward_file_name, delimiter=';')

        reward_pos_data = []
        reward_neg_data = []
        if mode == Mode.BOTH or mode == Mode.POSITIVE:
            reward_pos_data = genfromtxt(pos_reward_file_name, delimiter=';')
        if mode == Mode.BOTH or mode == Mode.NEGATIVE:
            reward_neg_data = genfromtxt(neg_reward_file_name, delimiter=';')

        return score_data, reward_data, reward_pos_data, reward_neg_data

