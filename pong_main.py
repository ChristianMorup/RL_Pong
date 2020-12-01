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
modes = all_modes

n_simulations = 10
frames_per_second = 20000
num_episodes = 1000
episode_length = 100
slow_down_on_last_100 = False

for reward_type in reward_types:
    for mode in modes:
        file_util = FileUtil('sim_data/rev3/')

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
            Q = defaultdict(lambda: np.zeros(nA))

            cum_score = []
            cum_reward = []
            cum_pos_reward = []
            cum_neg_reward = []

            # -------- Main Program Loop -----------
            for i_episode in range(1, num_episodes + 1):
                eps = 1.0 / i_episode
                tries = 0
                reward_sum = 0
                reward_pos_sum = 0
                reward_neg_sum = 0
                wall_hits = 0
                new_dist = 0
                done = False

                pygame.init()
                screen = pygame.display.set_mode(size)
                pygame.display.set_caption("Pong")

                paddleA = Paddle(WHITE, paddle_width, paddle_height)
                paddleA.rect.x = 0
                paddleA.rect.y = 0

                paddleB = Paddle(WHITE, paddle_width, paddle_height)
                paddleB.rect.x = x  # x-paddle_width
                paddleB.rect.y = 20

                ball = Ball(WHITE, ball_dim, ball_dim)
                ball.rect.x = 35
                ball.rect.y = 20

                state = ((ball.rect.x, ball.rect.y), (ball.rect.x, ball.rect.y), paddleA.rect.y)
                action = sarsa.epsilon_greedy(Q, state, nA, eps)

                # This will be a list that will contain all the sprites we intend to use in our game.
                all_sprites_list = pygame.sprite.Group()

                # Add the car to the list of objects
                all_sprites_list.add(paddleA)
                # all_sprites_list.add(paddleB)
                all_sprites_list.add(ball)

                # The clock will be used to control how fast the screen updates
                clock = pygame.time.Clock()

                rdm = RewardDecisionMaker(mode, reward_type)

                print(f'Episode {i_episode}')
                if slow_down_on_last_100 and i_episode > num_episodes-100:
                    frames_per_second = 60
                while True:
                    # --- Main event loop
                    curr_reward = 0
                    curr_pos_reward = 0
                    curr_neg_reward = 0

                    for event in pygame.event.get():  # User did something
                        if event.type == pygame.QUIT:  # If user clicked close
                            carryOn = False  # Flag that we are done so we exit this loop

                    # Moving the paddles when the use uses the arrow keys (player A) or "W/S" keys (player B)
                    moved = False
                    if action == 0:
                        paddleA.moveUp(5)
                        moved = True
                    if action == 1:
                        paddleA.moveDown(5)
                        moved = True

                        # --- Game logic should go here
                    all_sprites_list.update()
                    old_dist = new_dist
                    new_dist = abs((paddleA.rect.y + paddle_width // 2) - ball.rect.y)

                    paddleA_hit = pygame.sprite.collide_mask(ball, paddleA)
                    # paddleB_hit = pygame.sprite.collide_mask(ball, paddleB)
                    wall_hit = ball.rect.x < 0
                    # Detect collisions between the ball and the paddles
                    if paddleA_hit:
                        ball.bounce()
                        tries += 1

                    if not paddleA_hit:
                        # Check if the ball is bouncing against any of the 4 walls:
                        if ball.rect.x > x - ball_dim:
                            ball.velocity[0] = -ball.velocity[0]
                        if ball.rect.x < 0:
                            tries += 1
                            wall_hits += 1
                            # scoreA -= dist_between_paddle_ball // 20
                            ball.velocity[0] = -ball.velocity[0]
                        if ball.rect.y > y - 5 - ball_dim:
                            ball.velocity[1] = -ball.velocity[1]
                        if ball.rect.y < 5:
                            ball.velocity[1] = -ball.velocity[1]

                    curr_reward, curr_pos_reward, curr_neg_reward = rdm.calculate_rewards(old_dist=old_dist,
                                                                                          new_dist=new_dist,
                                                                                          paddleA_hit=paddleA_hit,
                                                                                          wall_hit=wall_hit,
                                                                                          dist_reward=0.1,
                                                                                          hit_reward=100,
                                                                                          ball_x=ball.rect.x,
                                                                                          moved=moved,
                                                                                          ball_velocity=ball.velocity[0]
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
                    reward = curr_reward
                    reward_sum += reward
                    reward_neg_sum += curr_neg_reward
                    reward_pos_sum += curr_pos_reward

                    if not done:
                        # Choose action A_{t+1} using policy derived from Q (e.g., eps-greedy)
                        next_state = ((ball.rect.x, ball.rect.y), (state[1][0], state[1][1]), paddleA.rect.y)
                        next_action = sarsa.epsilon_greedy(Q, next_state, nA, eps)
                        # Update Q
                        Q[state][action] = sarsa.update_Q_sarsa(0.5, 0.1, Q, state, action,
                                                                reward, next_state, next_action)
                        # Update state and action
                        state = next_state  # S_t <- S_{t+1}
                        action = next_action  # A_t <- A_{t+1}
                    if done:
                        Q[state][action] = sarsa.update_Q_sarsa(0.5, 0.1, Q,
                                                                state, action, reward)
                        cum_reward.append(reward_sum)
                        cum_score.append(wall_hits)
                        cum_neg_reward.append(reward_neg_sum)
                        cum_pos_reward.append(reward_pos_sum)
                        break

                    # --- Limit to 60 frames per second
                    clock.tick(frames_per_second)

            pygame.quit()
            sim_end = time()
            print(f'Total time: {sim_end - sim_start}s')

            file_util.write_to_csv(reward_type, mode, cum_score=cum_score, cum_reward=cum_reward,
                                   cum_pos_reward=cum_pos_reward, cum_neg_reward=cum_neg_reward)
