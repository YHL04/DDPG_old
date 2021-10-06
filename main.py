import gym
import numpy as np
import matplotlib.pyplot as plt
import time

from ddpg import Agent
from env import Market

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train = Market(training=True,
             initial_value=10000,
             upper_bound=10)
test = Market(training=False,
             initial_value=10000,
             upper_bound=10)

num_states = train.state_size
num_actions = train.action_size
upper_bound = train.upper_bound

print("Num States: ", num_states)
print("Num Actions: ", num_actions)
print("Max Value of Action: ", upper_bound)

agent = Agent(num_states, num_actions, upper_bound=upper_bound)
# log = open(f"logs/history.txt", "w")


def train_step(agent, env):
    done = False
    total_reward = 0
    state = env.reset()
    agent.reset_state()
    state = agent.process_state(state)

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = agent.process_state(next_state)

        total_reward += reward

        if env.training:
            agent.remember(state, action, reward, next_state, done)
            agent.learn()

        state = next_state

    if env.training:
        agent.update_target()

    return total_reward


avg_reward = []
ep_reward = []

ep_val_reward = []
avg_val_reward = []

start = time.time()

for i in range(100000):
    reward = train_step(agent, train)
    ep_reward.append(reward)
    avg_reward.append(np.mean(ep_reward[-20:]))

    val_reward = train_step(agent, test)
    ep_val_reward.append(val_reward)
    avg_val_reward.append(np.mean(ep_val_reward[-20:]))

    print(f"Episode {i} \t "
          f"Average Reward: {reward} \t "
          f"Val Reward: {val_reward}")
    print("Time: ", time.time()-start)
    # log.write(f"{i}, {reward}, {val_reward}\n")
    # log.flush()

    # if i % 1000 == 0:
    #     agent.save(i)


# log.close()
plt.subplot(4, 1, 1)
plt.plot(ep_reward)
plt.plot(avg_reward)
plt.subplot(4, 1, 2)
plt.plot(ep_val_reward)
plt.plot(avg_val_reward)

plt.show()