import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import numpy as np
import random
from collections import deque
import itertools

from utils import OUActionNoise, RunningMeanStd, normalize


def build_actor(num_states, num_actions, upper_bound):
    model = keras.Sequential(
        [
            layers.Input(shape=num_states,),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.LayerNormalization(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.LayerNormalization(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.LayerNormalization(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_actions, activation="tanh"),
            layers.Lambda(lambda x: x*upper_bound)
        ]
    )
    return model


def build_critic(num_states, num_actions):
    state_input  = layers.Input(shape=(num_states,))
    state_out    = layers.Dense(256, activation="relu")(state_input)
    state_out    = layers.BatchNormalization()(state_out)
    state_out    = layers.Dense(256, activation="relu")(state_out)

    action_input = layers.Input(shape=(num_actions,))
    action_out   = layers.Dense(64, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(1024, activation="relu")(concat)
    out = layers.Dense(512, activation="relu")(out)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(128, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = keras.Model([state_input, action_input], outputs)
    return model


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class Agent:
    def __init__(self, num_states, num_actions, upper_bound):
        self.num_states = num_states
        self.num_actions = num_actions
        self.upper_bound = upper_bound
        # self.stack_state = 200
        self.stack_state = 2000

        self.gamma = 0.99
        self.batch_size = 64

        self.actor = build_actor(num_states*self.stack_state, num_actions, upper_bound)
        self.critic = build_critic(num_states*self.stack_state, num_actions)
        self.target_actor = build_actor(num_states*self.stack_state, num_actions, upper_bound)
        self.target_critic = build_critic(num_states*self.stack_state, num_actions)
        self.update_target()

        self.actor_optimizer = keras.optimizers.Adam(0.0001)
        self.critic_optimizer = keras.optimizers.Adam(0.0002)

        # self.buffer_size = 400000
        self.buffer_size = 40000

        self.state_buffer = np.zeros((self.buffer_size, self.num_states*self.stack_state), dtype="float32")
        self.reward_buffer = np.zeros((self.buffer_size, 1), dtype="float32")
        self.action_buffer = np.zeros((self.buffer_size, self.num_actions), dtype="float32")
        self.next_state_buffer = np.zeros((self.buffer_size, self.num_states*self.stack_state), dtype="float32")

        self.index = 0
        self.ou_noise = OUActionNoise(mean=np.zeros(self.num_actions),
                                      std_deviation=0.1*self.upper_bound*np.ones(self.num_actions))

        self.state = deque(maxlen=self.stack_state)
        self.reset_state()

        self.epsilon = 1
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.2

    def reset_state(self):
        for i in range(self.stack_state):
            self.state.append([0]*self.num_states)

    def process_state(self, state):
        self.state.append(state)
        state = list(itertools.chain.from_iterable(self.state))
        return np.array(state)

    def process_reward(self, reward):
        return reward

    def get_action(self, state, training=True):
        if training and random.uniform(0, 1) <= self.epsilon:
            return np.random.uniform(-self.upper_bound, self.upper_bound, self.num_actions)
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        actions = tf.squeeze(self.actor(state))
        noise = self.ou_noise()
        if not training:
            noise *= 0
        actions = actions.numpy() + noise
        actions = np.clip(actions, -self.upper_bound, self.upper_bound)

        return actions

    def remember(self, state, action, reward, next_state, done):
        pointer = self.index % self.buffer_size

        self.state_buffer[pointer] = state
        self.action_buffer[pointer] = action
        self.reward_buffer[pointer] = reward
        self.next_state_buffer[pointer] = next_state

        self.index += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self):
        record_range = min(self.index, self.buffer_size)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        actor_loss, critic_loss = self.update(state_batch, action_batch, reward_batch, next_state_batch)

        return actor_loss, critic_loss

    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch
    ):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

        return actor_loss, critic_loss

    def update_target(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def update_target_slow(self):
        update_target(self.target_actor.variables, self.actor.variables, 0.005)
        update_target(self.target_critic.variables, self.critic.variables, 0.005)

    def save(self, name):
        self.actor.save_weights(f"model/{name}/actor")
        self.critic.save_weights(f"model/{name}/critic")

    def load(self, name):
        self.actor.load_weights(f"model/{name}/actor")
        self.critic.load_weights(f"model/{name}/critic")
