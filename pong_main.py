# Import the pygame library and initialise the game engine
import pygame
from paddle import Paddle
from ball import Ball
import sys
import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random

pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Open a new window
x = 140
y = 100
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
all_sprites_list.add(paddleB)
all_sprites_list.add(ball)

# The loop will carry on until the user exit the game (e.g. clicks the close button).
carryOn = True

# The clock will be used to control how fast the screen updates
clock = pygame.time.Clock()

# Initialise player scores
scoreA = 0
scoreB = 0


# SARSA
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


nA = 3  # Up, Down, Do-nothing
Q = defaultdict(lambda: np.zeros(nA))
state = ((ball.rect.x, ball.rect.y), (ball.rect.x, ball.rect.y), paddleA.rect.x, paddleA.rect.y)
num_episodes = 10


def sarsa(num_episodes, alpha, gamma=1.0):
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

            ## ---> COMPLETE THE FUNCTION <---
        # Set epsilon
        eps = 1.0 / i_episode
        # Initialize score
        score = 0
        # Observe S_0: the initial state is #36
        # state = env.reset()
        # Choose action A_0 using policy derived from Q (e.g., eps-greedy)
        action = epsilon_greedy(Q, state, nA, eps)

        while True:
            next_state, reward, done, info = env.step(action)  # take action A, observe R', S'
            score += reward  # add reward to agent's score
            if not done:
                # Choose action A_{t+1} using policy derived from Q (e.g., eps-greedy)
                next_action = epsilon_greedy(Q, next_state, nA, eps)
                # Update Q
                Q[state][action] = update_Q_sarsa(alpha, gamma, Q, state, action,
                                                  reward, next_state, next_action)
                # Update state and action
                state = next_state  # S_t <- S_{t+1}
                action = next_action  # A_t <- A_{t+1}
            if done:
                Q[state][action] = update_Q_sarsa(alpha, gamma, Q,
                                                  state, action, reward)
                break

    return Q


# -------- Main Program Loop -----------
for i_episode in range(1, num_episodes + 1):

    eps = 1.0 / i_episode
    scoreA = 0
    dist_between_paddle_ball = 0
    action = epsilon_greedy(Q, state, nA, eps)
    print(f'Episode {i_episode}')
    frames_per_second = 10000
    if i_episode > 9:
        frames_per_second = 60
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
        # Detect collisions between the ball and the paddles
        if paddleA_hit:
            #scoreA += 20
            #print(scoreA)
            ball.bounce()
        if paddleB_hit:
            ball.bounce()

        if not (paddleA_hit or paddleB_hit):
            # Check if the ball is bouncing against any of the 4 walls:
            if ball.rect.x > x - ball_dim:
                scoreA += 1
                ball.velocity[0] = -ball.velocity[0]
            if ball.rect.x < 0:
                scoreB += 1
                #scoreA -= dist_between_paddle_ball // 20
                ball.velocity[0] = -ball.velocity[0]
            if ball.rect.y > y - ball_dim:
                ball.velocity[1] = -ball.velocity[1]
            if ball.rect.y < 0:
                ball.velocity[1] = -ball.velocity[1]

        if old_dist_between_paddle_ball-dist_between_paddle_ball > 0:
            curr_reward += 1
        else:
            curr_reward -= 1

        done = False
        if scoreA > 30 or scoreA < -30:
            done = True

        # --- Drawing code should go here
        # First, clear the screen to black.
        screen.fill(BLACK)
        # Draw the net
        pygame.draw.line(screen, WHITE, [x // 2, 0], [x // 2, y], 5)

        # Now let's draw all the sprites in one go. (For now we only have 2 sprites!)
        all_sprites_list.draw(screen)

        # Display scores:
        # font = pygame.font.Font(None, 74)
        # text = font.render(str(scoreA), 1, WHITE)
        # screen.blit(text, (25, 1))
        # text = font.render(str(scoreB), 1, WHITE)
        # screen.blit(text, (42, 1))

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
        reward = curr_reward
        if not done:
            # Choose action A_{t+1} using policy derived from Q (e.g., eps-greedy)
            next_state = ((ball.rect.x, ball.rect.y), (state[1][0], state[1][1]), paddleA.rect.x, paddleA.rect.y)
            next_action = epsilon_greedy(Q, next_state, nA, eps)
            # Update Q
            Q[state][action] = update_Q_sarsa(500, 0.1, Q, state, action,
                                              reward, next_state, next_action)
            # Update state and action
            state = next_state  # S_t <- S_{t+1}
            action = next_action  # A_t <- A_{t+1}
        if done:
            Q[state][action] = update_Q_sarsa(500, 0.1, Q,
                                              state, action, reward)
            break

        # --- Limit to 60 frames per second
        clock.tick(frames_per_second)


#print(Q)

# Once we have exited the main program loop we can stop the game engine:
pygame.quit()
