from enum import Enum


class RewardType(Enum):
    TRACKING = 1
    HITTING = 2
    TRACKING_AND_HITTING = 3
    TRACKING_PROPORTIONAL = 4
    TRACKING_PROPORTIONAL_UNIDIRECTIONAL = 5
    TRACKING_PROPORTIONAL_UNIDIRECTIONAL_WEIGHTED = 6
    TRACKING_STEADY_NERVES = 7


class Mode(Enum):
    BOTH = 1
    NEGATIVE = 2
    POSITIVE = 3


all_modes = [Mode.BOTH, Mode.NEGATIVE, Mode.POSITIVE]

mode_map = {Mode.BOTH: 'both', Mode.NEGATIVE: 'negative', Mode.POSITIVE: 'positive'}

reward_type_map = {RewardType.TRACKING: 'tracking',
                   RewardType.HITTING: 'hitting',
                   RewardType.TRACKING_AND_HITTING: 'tracking_and_hitting',
                   RewardType.TRACKING_PROPORTIONAL: 'tracking_proportional',
                   RewardType.TRACKING_PROPORTIONAL_UNIDIRECTIONAL: 'tracking_proportional_uni',
                   RewardType.TRACKING_PROPORTIONAL_UNIDIRECTIONAL_WEIGHTED: 'tracking_proportional_uni_weighted',
                   RewardType.TRACKING_STEADY_NERVES: 'tracking_steady_nerves'}

all_rewards = [RewardType.TRACKING, RewardType.HITTING, RewardType.TRACKING_AND_HITTING, RewardType.TRACKING_PROPORTIONAL,
           RewardType.TRACKING_PROPORTIONAL_UNIDIRECTIONAL, RewardType.TRACKING_PROPORTIONAL_UNIDIRECTIONAL_WEIGHTED,
           RewardType.TRACKING_STEADY_NERVES]

