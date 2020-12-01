from modes_and_types import Mode
from modes_and_types import RewardType


class RewardDecisionMaker:
    mode = None
    reward_type = None

    def __init__(self, mode, reward_type):
        self.mode = mode
        self.reward_type = reward_type

    def calculate_rewards(self, **kwargs):
        if self.reward_type == RewardType.TRACKING:
            return self.reward_for_tracking(kwargs.get('old_dist'), kwargs.get('new_dist'))
        elif self.reward_type == RewardType.HITTING:
            return self.reward_for_hitting(kwargs.get('paddleA_hit'), kwargs.get('wall_hit'))
        elif self.reward_type == RewardType.TRACKING_AND_HITTING:
            return self.reward_for_tracking_and_hitting(kwargs.get('old_dist'),
                                                        kwargs.get('new_dist'),
                                                        kwargs.get('paddleA_hit'),
                                                        kwargs.get('wall_hit'),
                                                        kwargs.get('dist_reward'),
                                                        kwargs.get('hit_reward'))
        elif self.reward_type == RewardType.TRACKING_PROPORTIONAL:
            return self.reward_for_tracking_proportional(kwargs.get('old_dist'), kwargs.get('new_dist'))
        elif self.reward_type == RewardType.TRACKING_PROPORTIONAL_UNIDIRECTIONAL:
            return self.reward_for_tracking_proportional_unidirectional(kwargs.get('old_dist'),
                                                                        kwargs.get('new_dist'),
                                                                        kwargs.get('ball_velocity'))
        elif self.reward_type == RewardType.TRACKING_PROPORTIONAL_UNIDIRECTIONAL_WEIGHTED:
            return self.reward_for_tracking_proportional_unidirectional_weighted(kwargs.get('old_dist'),
                                                                        kwargs.get('new_dist'),
                                                                        kwargs.get('ball_x'),
                                                                        kwargs.get('ball_velocity'))
        elif self.reward_type == RewardType.TRACKING_STEADY_NERVES:
            return self.reward_for_tracking_and_standing_still(kwargs.get('old_dist'),
                                                                        kwargs.get('new_dist'),
                                                                        kwargs.get('ball_velocity'),
                                                                        kwargs.get('moved'))
        else:
            return 0, 0, 0

    def reward_for_tracking(self, old_dist, new_dist, dist_reward=1):
        reward = 0
        reward_pos = 0
        reward_neg = 0

        if old_dist - new_dist >= 0:
            if self.mode == Mode.BOTH or self.mode == Mode.POSITIVE:
                reward += dist_reward
                reward_pos += dist_reward
        elif old_dist - new_dist < 0:
            if self.mode == Mode.BOTH or self.mode == Mode.NEGATIVE:
                reward -= dist_reward
                reward_neg -= dist_reward

        return reward, reward_pos, reward_neg

    def reward_for_hitting(self, paddleA_hit, wall_hit, hit_reward=1):
        reward = 0
        reward_pos = 0
        reward_neg = 0
        if paddleA_hit:
            if self.mode == Mode.BOTH or self.mode == Mode.POSITIVE:
                reward += hit_reward
                reward_pos += hit_reward
        elif wall_hit:
            if self.mode == Mode.BOTH or self.mode == Mode.NEGATIVE:
                reward -= hit_reward
                reward_neg -= hit_reward

        return reward, reward_pos, reward_neg

    def reward_for_tracking_and_hitting(self, old_dist, new_dist, paddleA_hit, wall_hit, dist_reward, hit_reward):
        r_dist, r_pos_dist, r_neg_dist = self.reward_for_tracking(old_dist, new_dist, dist_reward=dist_reward)
        r_hit, r_pos_hit, r_neg_hit = self.reward_for_hitting(paddleA_hit, wall_hit, hit_reward=hit_reward)

        return r_dist + r_hit, r_pos_dist + r_pos_hit, r_neg_dist + r_neg_hit

    def reward_for_tracking_proportional(self, old_dist, new_dist):
        return self.reward_for_tracking(old_dist, new_dist, dist_reward=abs(old_dist - new_dist))

    def reward_for_tracking_proportional_unidirectional(self, old_dist, new_dist, ball_velocity):
        if ball_velocity < 0:
            return self.reward_for_tracking(old_dist, new_dist, dist_reward=abs(old_dist - new_dist))
        else:
            return 0, 0, 0

    def reward_for_tracking_proportional_unidirectional_weighted(self, old_dist, new_dist, ball_x, ball_velocity):
        if ball_velocity < 0:
            return self.reward_for_tracking(old_dist, new_dist, dist_reward=abs(old_dist - new_dist) * (140 - ball_x))
        else:
            return 0, 0, 0

    def reward_for_tracking_and_standing_still(self, old_dist, new_dist, ball_velocity, moved):
        r_dist, r_pos_dist, r_neg_dist, r_move, r_pos_move, r_neg_move = 0, 0, 0, 0, 0, 0

        if ball_velocity < 0:
            r_dist, r_pos_dist, r_neg_dist = self.reward_for_tracking(old_dist, new_dist,
                                                                      dist_reward=abs(old_dist - new_dist))
        else:
            if moved:
                if self.mode == Mode.BOTH or self.mode == Mode.NEGATIVE:
                    r_move -= 0.5
                    r_neg_move -= 0.5
            elif not moved:
                if self.mode == Mode.BOTH or self.mode == Mode.POSITIVE:
                    r_move += 0.5
                    r_neg_move += 0.5

        return r_dist + r_move, r_pos_dist + r_pos_move, r_neg_dist + r_neg_move
