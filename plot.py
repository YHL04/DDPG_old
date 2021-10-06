import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from env import Market, get_data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data = pd.read_csv("logs/history.txt",
                   names=["Episode", "Train", "Test"])

prices = get_data("data.pkl", False)

print(prices)
print(prices.iloc[0])
print(prices.iloc[-1])
print((prices.iloc[-1] / prices.iloc[0]) * 10000)
print(((prices.iloc[-1] / prices.iloc[0]) * 10000).mean())

def random_actions(mode="train"):
    upper_bound = 10
    if mode == "train":
        env = Market(training=True,
                     initial_value=10000,
                     upper_bound=upper_bound)
    if mode == "test":
        env = Market(training=False,
                     initial_value=10000,
                     upper_bound=upper_bound)


    done = False
    total_reward = 0
    state = env.reset()

    while not done:
        action = np.random.uniform(-upper_bound, upper_bound, env.action_size)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

    return total_reward


plt.subplot(2, 1, 1)
#plt.plot(data["Train"])
#plt.plot(data["Train"].rolling(20).mean())
plt.plot(data["Train"].rolling(200).mean())

plt.title('200 Rolling Average training profit over epochs ($)')

# random_list = [random_actions(mode="train") for _ in range(10)]
# average_reward = sum(random_list) / len(random_list)
#
# plt.axhline(y=max(random_list), color='orange')
# plt.axhline(y=min(random_list), color='orange')
# plt.axhline(y=average_reward, color='r')

plt.subplot(2, 1, 2)
#plt.plot(data["Test"])
#plt.plot(data["Test"].rolling(20).mean())
plt.plot(data["Test"].rolling(200).mean())

plt.title("200 Rolling Average testing profit over epochs ($)")

# random_list = [random_actions(mode="test") for _ in range(10)]
# average_reward = sum(random_list) / len(random_list)
#
# plt.axhline(y=max(random_list), color='orange')
# plt.axhline(y=min(random_list), color='orange')
# plt.axhline(y=average_reward, color='r')

plt.show()
