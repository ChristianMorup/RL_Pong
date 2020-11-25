# Import the pygame library and initialise the game engine
import csv
from collections import defaultdict

import numpy as np
import pygame

import reward_decision_maker as rdm
import sarsa
from ball import Ball
from paddle import Paddle

reward_types = ['tracking', 'hitting_the_ball', 'combo']
modes = ['both', 'negative', 'positive']
n_simulations = 15
frames_per_second = 20000
num_episodes = 1000

for reward_type in reward_types:
    for mode in modes:
        score_file_name = 'Simulation_' + reward_type + '_' + mode + '_score.csv'
        reward_file_name = 'Simulation_' + reward_type + '_' + mode + '_reward.csv'

        for i_simulation in range(1, n_simulations + 1):
            print(f'Simulation: {i_simulation}')
            pygame.init()

            # Define some colors
            BLACK = (0, 0, 0)
            WHITE = (255, 255, 255)

            # Open a new window
            x = 140
            y = 200
            paddle_width = 5
            paddle_height = 20
            ball_dim = 5
            size = (x, y)
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

            # This will be a list that will contain all the sprites we intend to use in our game.
            all_sprites_list = pygame.sprite.Group()

            # Add the car to the list of objects
            all_sprites_list.add(paddleA)
            # all_sprites_list.add(paddleB)
            all_sprites_list.add(ball)

            # The clock will be used to control how fast the screen updates
            clock = pygame.time.Clock()

            nA = 3  # Up, Down, Do-nothing
            Q = defaultdict(lambda: np.zeros(nA))
            state = ((ball.rect.x, ball.rect.y), (ball.rect.x, ball.rect.y), paddleA.rect.x, paddleA.rect.y)

            cum_score = []
            cum_reward = []

            # -------- Main Program Loop -----------
            for i_episode in range(1, num_episodes + 1):
                eps = 1.0 / i_episode
                scoreA = 0
                scoreB = 0
                tries = 0
                reward_sum = 0
                wall_hits = 0
                dist_between_paddle_ball = 0
                done = False
                action = sarsa.epsilon_greedy(Q, state, nA, eps)

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

                # This will be a list that will contain all the sprites we intend to use in our game.
                all_sprites_list = pygame.sprite.Group()

                # Add the car to the list of objects
                all_sprites_list.add(paddleA)
                # all_sprites_list.add(paddleB)
                all_sprites_list.add(ball)

                # The clock will be used to control how fast the screen updates
                clock = pygame.time.Clock()


                print(f'Episode {i_episode}')
                # if i_episode > 900:
                #     frames_per_second = 60
                while True:
                    # --- Main event loop
                    curr_reward = 0

                    for event in pygame.event.get():  # User did something
                        if event.type == pygame.QUIT:  # If user clicked close
                            carryOn = False  # Flag that we are done so we exit this loop
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_x:  # Pressing the x Key will quit the game
                                carryOn = False

                    # Moving the paddles when the use uses the arrow keys (player A) or "W/S" keys (player B)
                    keys = pygame.key.get_pressed()
                    if action == 0:
                        paddleA.moveUp(5)
                    if action == 1:
                        paddleA.moveDown(5)

                        # --- Game logic should go here
                    all_sprites_list.update()
                    old_dist_between_paddle_ball = dist_between_paddle_ball
                    dist_between_paddle_ball = abs((paddleA.rect.y + paddle_width // 2) - ball.rect.y)

                    paddleA_hit = pygame.sprite.collide_mask(ball, paddleA)
                    paddleB_hit = pygame.sprite.collide_mask(ball, paddleB)
                    wall_hit = ball.rect.x < 0
                    # Detect collisions between the ball and the paddles
                    if paddleA_hit:
                        ball.bounce()
                        tries += 1
                    if paddleB_hit:
                        ball.bounce()

                    if not (paddleA_hit or paddleB_hit):
                        # Check if the ball is bouncing against any of the 4 walls:
                        if ball.rect.x > x - ball_dim:
                            scoreA += 1
                            tries += 1
                            ball.velocity[0] = -ball.velocity[0]
                        if ball.rect.x < 0:
                            scoreB += 1
                            wall_hits += 1
                            # scoreA -= dist_between_paddle_ball // 20
                            ball.velocity[0] = -ball.velocity[0]
                        if ball.rect.y > y - ball_dim:
                            ball.velocity[1] = -ball.velocity[1]
                        if ball.rect.y < 0:
                            ball.velocity[1] = -ball.velocity[1]

                    if reward_type == 'tracking':
                        curr_reward = rdm.reward_for_tracking_the_ball(old_dist_between_paddle_ball, dist_between_paddle_ball, mode)
                    elif reward_type == 'hitting_the_ball':
                        curr_reward = rdm.reward_for_hitting_the_ball(paddleA_hit, wall_hit, mode)
                    else:
                        curr_reward = rdm.reward_for_tracking_the_ball(old_dist_between_paddle_ball, dist_between_paddle_ball, mode)
                        curr_reward += rdm.reward_for_hitting_the_ball(paddleA_hit, wall_hit, mode)

                    if tries > 100:
                        done = True

                    # --- Drawing code should go here
                    # First, clear the screen to black.
                    screen.fill(BLACK)
                    # Draw the net
                    pygame.draw.line(screen, WHITE, [x // 2, 0], [x // 2, y], 5)

                    # Now let's draw all the sprites in one go. (For now we only have 2 sprites!)
                    all_sprites_list.draw(screen)

                    # --- Go ahead and update the screen with what we've drawn.
                    pygame.display.flip()
                    reward = curr_reward
                    reward_sum += reward
                    if not done:
                        # Choose action A_{t+1} using policy derived from Q (e.g., eps-greedy)
                        next_state = ((ball.rect.x, ball.rect.y), (state[1][0], state[1][1]), paddleA.rect.x, paddleA.rect.y)
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
                        break

                    # --- Limit to 60 frames per second
                    clock.tick(frames_per_second)

            pygame.quit()

            f = open(score_file_name, 'a')
            csv.writer(f, delimiter=';').writerow(cum_score)
            f.close()

            f = open(reward_file_name, 'a')
            csv.writer(f, delimiter=';').writerow(cum_reward)
            f.close()





# filtered_cum_score = []
# filtered_cum_reward = []
# for i in range(0, len(cum_score) - 10):
#     sum_score = 0
#     sum_reward = 0
#     for j in range(0, 10):
#         sum_score += cum_score[i + j]
#         sum_reward += cum_reward[i + j]
#     average_score = sum_score / 10
#     average_reward = sum_reward / 10
#     filtered_cum_score.append(average_score)
#     filtered_cum_reward.append(average_reward)
#
# plt.plot(range(0, num_episodes - 10), filtered_cum_score)
# plt.ylabel('Number of \'back wall\'-hits')
# plt.xlabel('Episodes')
# plt.title('Score - ' + mode)
# plt.show()
# plt.clf()
# plt.plot(range(0, num_episodes - 10), filtered_cum_reward)
# plt.ylabel('Cumulative reward')
# plt.xlabel('Episodes')
# plt.title('Reward - ' + mode)
# plt.show()

