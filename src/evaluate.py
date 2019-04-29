import gym
import numpy as np
import time

from environment import GoEnv, make_random_policy


def evaluate_baseline(num_episode=1, max_time_step_per_episode=1000, verbose=False):
    env = GoEnv(player_color='black', illegal_move_mode='raise', board_size=9)
    random_policy = make_random_policy(env.np_random)
    total_rewards = []
    for i_episode in range(num_episode):
        total_reward = 0.0
        observation = env.reset()
        for t in range(max_time_step_per_episode):
            if verbose:
                env.render()
                print(observation)
            action = random_policy(env.state, None, None)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print(f"[{i_episode}]: t={t+1}, total_reward={total_reward}")
                env.render()
                break
        total_rewards.append(total_reward)
    env.close()
    return total_rewards


num_episods = 10
t0 = time.time()
total_rewards = evaluate_baseline(num_episode=num_episods)
wins = [int(r == 1.0) for r in total_rewards]
t1 = time.time()
print(f'win: {sum(wins)} out of {len(wins)} games, {(t1-t0)/num_episods:.1f}s per game')
