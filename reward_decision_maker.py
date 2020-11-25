def reward_for_tracking_the_ball(old_dist, new_dis, mode='both'):
    if mode == 'both':
        if old_dist - new_dis >= 0:
            return 1
        else:
            return -1
    elif mode == 'positive':
        if old_dist - new_dis > 0:
            return 1
        else:
            return 0
    elif mode == 'negative':
        if old_dist - new_dis > 0:
            return 0
        else:
            return -1
    return 0


def reward_for_hitting_the_ball(paddleA_hit, wall_hit, mode='both'):
    if mode == 'both':
        if paddleA_hit:
            return 1
        elif wall_hit:
            return -1
        return 0
    elif mode == 'positive':
        if paddleA_hit:
            return 1
        return 0
    elif mode == 'negative':
        if wall_hit:
            return -1
        return 0
    return 0