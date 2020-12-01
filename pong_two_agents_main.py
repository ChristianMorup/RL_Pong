from collections import defaultdict
import numpy as np
import pygame
from reward_decision_maker import RewardDecisionMaker
import sarsa
from ball import Ball
from paddle import Paddle
from time import time
from modes_and_types import *
from file_util import FileUtil

# reward_types = all_rewards
reward_types = [RewardType.TRACKING]
modeA = Mode.BOTH
modeB = Mode.BOTH
reward_typeA = RewardType.TRACKING_PROPORTIONAL_UNIDIRECTIONAL
reward_typeB = RewardType.TRACKING_PROPORTIONAL_UNIDIRECTIONAL_WEIGHTED

n_simulations = 5
frames_per_second = 20000
num_episodes = 1000
episode_length = 100
slow_down_on_last_100 = False

file_util = FileUtil('sim_data/test/')

for i_simulation in range(1, n_simulations + 1):
    sim_start = time()
    print(f'Simulation: {i_simulation}')
    pygame.init()

    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    # Open a new window
    x = 140
    y = 150
    paddle_width = 5
    paddle_height = 20
    ball_dim = 5
    size = (x, y)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Pong")

    nA = 3  # Up, Down, Do-nothing
    Q_a = defaultdict(lambda: np.zeros(nA))
    Q_b = defaultdict(lambda: np.zeros(nA))

    a_cum_score = []
    a_cum_reward = []
    a_cum_pos_reward = []
    a_cum_neg_reward = []

    b_cum_score = []
    b_cum_reward = []
    b_cum_pos_reward = []
    b_cum_neg_reward = []

    # -------- Main Program Loop -----------
    for i_episode in range(1, num_episodes + 1):
        eps = 1.0 / i_episode
        tries = 0
        a_reward_sum = 0
        a_reward_pos_sum = 0
        a_reward_neg_sum = 0
        b_reward_sum = 0
        b_reward_pos_sum = 0
        b_reward_neg_sum = 0
        a_wall_hits = 0
        b_wall_hits = 0
        a_new_dist = 0
        b_new_dist = 0
        done = False

        pygame.init()
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Pong")

        paddleA = Paddle(WHITE, paddle_width, paddle_height)
        paddleA.rect.x = 0
        paddleA.rect.y = 0

        paddleB = Paddle(WHITE, paddle_width, paddle_height)
        paddleB.rect.x = x - paddle_width
        paddleB.rect.y = 0

        ball = Ball(WHITE, ball_dim, ball_dim)
        ball.rect.x = 35
        ball.rect.y = 20

        state = ((ball.rect.x, ball.rect.y), (ball.rect.x, ball.rect.y), paddleA.rect.y, paddleB.rect.y)

        action_agent_a = sarsa.epsilon_greedy(Q_a, state, nA, eps)
        action_agent_b = sarsa.epsilon_greedy(Q_b, state, nA, eps)

        # This will be a list that will contain all the sprites we intend to use in our game.
        all_sprites_list = pygame.sprite.Group()

        # Add the car to the list of objects
        all_sprites_list.add(paddleA)
        all_sprites_list.add(paddleB)
        all_sprites_list.add(ball)

        # The clock will be used to control how fast the screen updates
        clock = pygame.time.Clock()

        rdmA = RewardDecisionMaker(modeA, reward_typeA)
        rdmB = RewardDecisionMaker(modeB, reward_typeB)

        print(f'Episode {i_episode}')
        if slow_down_on_last_100 and i_episode > num_episodes - 100:
            frames_per_second = 60
        while True:
            # --- Main event loop
            a_curr_reward = 0
            a_curr_pos_reward = 0
            a_curr_neg_reward = 0

            b_curr_reward = 0
            b_curr_pos_reward = 0
            b_curr_neg_reward = 0

            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    carryOn = False  # Flag that we are done so we exit this loop

            # Moving the paddles when the use uses the arrow keys (player A) or "W/S" keys (player B)
            moved_A = False
            if action_agent_a == 0:
                paddleA.moveUp(5)
                moved_A = True
            if action_agent_a == 1:
                paddleA.moveDown(5)
                moved_A = True

            moved_B = False
            if action_agent_b == 0:
                paddleB.moveUp(5)
                moved_B = True
            if action_agent_b == 1:
                paddleB.moveDown(5)
                moved_B = True

                # --- Game logic should go here
            all_sprites_list.update()
            a_old_dist = a_new_dist
            a_new_dist = abs((paddleA.rect.y + paddle_width // 2) - ball.rect.y)

            b_old_dist = b_new_dist
            b_new_dist = abs((paddleB.rect.y + paddle_width // 2) - ball.rect.y)

            paddleA_hit = pygame.sprite.collide_mask(ball, paddleA)
            paddleB_hit = pygame.sprite.collide_mask(ball, paddleB)
            wall_hit_a = ball.rect.x < 0
            wall_hit_b = ball.rect.x > x - ball_dim

            # Detect collisions between the ball and the paddles
            if paddleA_hit:
                ball.bounce()
                tries += 1
                wall_hit_a = False

            if paddleB_hit:
                ball.bounce()
                wall_hit_b = False

            if not (paddleA_hit or paddleB_hit):
                # Check if the ball is bouncing against any of the 4 walls:
                if ball.rect.x > x - ball_dim:
                    ball.velocity[0] = -ball.velocity[0]
                    b_wall_hits += 1
                if ball.rect.x < 0:
                    tries += 1
                    a_wall_hits += 1
                    # scoreA -= dist_between_paddle_ball // 20
                    ball.velocity[0] = -ball.velocity[0]
                if ball.rect.y > y - 5 - ball_dim:
                    ball.velocity[1] = -ball.velocity[1]
                if ball.rect.y < 5:
                    ball.velocity[1] = -ball.velocity[1]

            a_curr_reward, a_curr_pos_reward, a_curr_neg_reward = rdmA.calculate_rewards(old_dist=a_old_dist,
                                                                                         new_dist=a_new_dist,
                                                                                         paddleA_hit=paddleA_hit,
                                                                                         wall_hit=wall_hit_a,
                                                                                         dist_reward=0.1,
                                                                                         hit_reward=100,
                                                                                         ball_x=ball.rect.x,
                                                                                         moved=moved_A,
                                                                                         ball_velocity=ball.velocity[0]
                                                                                         )

            b_curr_reward, b_curr_pos_reward, b_curr_neg_reward = rdmB.calculate_rewards(old_dist=b_old_dist,
                                                                                         new_dist=b_new_dist,
                                                                                         paddleA_hit=paddleB_hit,
                                                                                         wall_hit=wall_hit_b,
                                                                                         dist_reward=0.1,
                                                                                         hit_reward=100,
                                                                                         ball_x=ball.rect.x,
                                                                                         moved=moved_B,
                                                                                         ball_velocity=-ball.velocity[0]
                                                                                         )

            if tries > episode_length:
                done = True

            # --- Drawing code should go here
            # First, clear the screen to black.
            screen.fill(BLACK)
            # Draw the net
            # pygame.draw.line(screen, WHITE, [x // 2, 0], [x // 2, y], 5)

            # Now let's draw all the sprites in one go. (For now we only have 2 sprites!)
            all_sprites_list.draw(screen)

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()
            a_reward = a_curr_reward
            a_reward_sum += a_reward
            a_reward_neg_sum += a_curr_neg_reward
            a_reward_pos_sum += a_curr_pos_reward

            b_reward = b_curr_reward
            b_reward_sum += b_reward
            b_reward_neg_sum += b_curr_neg_reward
            b_reward_pos_sum += b_curr_pos_reward

            if not done:
                # Choose action A_{t+1} using policy derived from Q (e.g., eps-greedy)
                next_state = (
                    (ball.rect.x, ball.rect.y), (state[1][0], state[1][1]), paddleA.rect.y, paddleB.rect.y)
                next_action_a = sarsa.epsilon_greedy(Q_a, next_state, nA, eps)
                next_action_b = sarsa.epsilon_greedy(Q_b, next_state, nA, eps)

                # Update Q
                Q_a[state][action_agent_a] = sarsa.update_Q_sarsa(0.5, 0.1, Q_a, state, action_agent_a,
                                                                  a_reward, next_state, next_action_a)
                Q_b[state][action_agent_b] = sarsa.update_Q_sarsa(0.5, 0.1, Q_b, state, action_agent_b,
                                                                  b_reward, next_state, next_action_b)
                # Update state and action
                state = next_state  # S_t <- S_{t+1}
                action_agent_a = next_action_a  # A_t <- A_{t+1}
                action_agent_b = next_action_b  # A_t <- A_{t+1}

            if done:
                Q_a[state][action_agent_a] = sarsa.update_Q_sarsa(0.5, 0.1, Q_a,
                                                                  state, action_agent_a, a_reward)
                a_cum_reward.append(a_reward_sum)
                a_cum_score.append(a_wall_hits)
                a_cum_neg_reward.append(a_reward_neg_sum)
                a_cum_pos_reward.append(a_reward_pos_sum)

                Q_b[state][action_agent_b] = sarsa.update_Q_sarsa(0.5, 0.1, Q_b,
                                                                  state, action_agent_b, b_reward)
                b_cum_reward.append(b_reward_sum)
                b_cum_score.append(b_wall_hits)
                b_cum_neg_reward.append(b_reward_neg_sum)
                b_cum_pos_reward.append(b_reward_pos_sum)
                break

            # --- Limit to 60 frames per second
            clock.tick(frames_per_second)

    pygame.quit()
    sim_end = time()
    print(f'Total time: {sim_end - sim_start}s')

    file_util.write_to_csv(reward_typeA, modeA, cum_score=a_cum_score, cum_reward=a_cum_reward,
                           cum_pos_reward=a_cum_pos_reward, cum_neg_reward=a_cum_neg_reward)

    file_util.write_to_csv(reward_typeB, modeB, cum_score=b_cum_score, cum_reward=b_cum_reward,
                           cum_pos_reward=b_cum_pos_reward, cum_neg_reward=b_cum_neg_reward)
