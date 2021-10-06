
import pandas as pd
import datetime
import os
import numpy as np
import random


train_test_split = "2020-01-01"


def get_data(filename, training):
    data = pd.read_pickle(filename)
    data = data.dropna()
    data = data[(data != 0).all(1)]
    data = data.iloc[:,
           data.columns.get_level_values(1)=="close"]

    if training:
        data = data.loc[:train_test_split]
    else:
        data = data.loc[train_test_split:]

    #data = data.head(50)
    return data


class Market:
    def __init__(self, training,
                 initial_value=10000,
                 upper_bound=10):
        self.training = training
        self.prices = get_data("data.pkl", training)
        self.prices.columns = self.prices.columns.get_level_values(0)

        self.index = self.prices.index.tolist()

        self.initial_value = initial_value
        self.time = 0
        self.num_stocks = len(self.prices.columns)
        self.max_timestep = len(self.index)

        self.state = []

        print("Starting: ", self.index[0].strftime("%Y-%m-%d"))
        print("Ending: ", self.index[-1].strftime("%Y-%m-%d"))
        print("Num Steps: ", self.max_timestep)

        self.state_size = 1 + 2*self.num_stocks
        self.action_size = self.num_stocks
        self.upper_bound = upper_bound


    def reset(self):
        self.time = 0
        prices = self.prices.loc[self.index[self.time]].tolist()

        self.state = [self.initial_value] + \
                     prices + \
                     [0 for _ in range(self.num_stocks)]
        return np.array(self.state)

    def step(self, actions):
        '''
        :param actions: a list of integers representing the changes to each stock
        '''
        if self.time >= self.max_timestep-1:
            return np.array(self.state), 0, True, {}

        actions = np.squeeze(np.round(actions))

        # perform actions
        start_total_asset = self.state[0] + sum(np.array(self.state[1:1+self.num_stocks])*np.array(self.state[1+self.num_stocks:]))
        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            self.sell_stock(index, actions[index])

        for index in buy_index:
            self.buy_stock(index, actions[index])

        self.time += 1
        self.state = [self.state[0]] + self.prices.loc[self.index[self.time]].tolist() + self.state[1+self.num_stocks:]
        end_total_asset = self.state[0] + sum(np.array(self.state[1:1+self.num_stocks])*np.array(self.state[1+self.num_stocks:]))

        reward = end_total_asset - start_total_asset
        return np.array(self.state), reward, False, {}

    def buy_stock(self, index, action):
        available_amount = self.state[0] // self.state[index+1]
        self.state[0] -= self.state[index+1]*min(available_amount, action)
        self.state[index+self.num_stocks+1] += min(available_amount, action)

    def sell_stock(self, index, action):
        if self.state[index+self.num_stocks+1] > 0:
            self.state[0] += self.state[index+1] \
                             * min(abs(action), self.state[index+self.num_stocks+1])
            self.state[index+self.num_stocks+1] -= \
                min(abs(action), self.state[index+self.num_stocks+1])

    def render(self):
        print(self.state)
        return np.array(self.state)

if __name__ == "__main__":
    market = Market(True)
    print(market.prices)
    #print(market.index)
    print(market.num_stocks)
    print(market.max_timestep)