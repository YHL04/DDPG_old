import numpy as np


class Memory(object):
    """
    self.reset() has to be called first before any other function
    self.append() has to be called first before retrieve

    Precondition:
    """

    def __init__(self, total_timestep, num_states, num_actions, num_stack, batch_size, buffer_size=300):
        self.buffer_size = buffer_size
        self.total_timestep = total_timestep
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_stack = num_stack
        self.batch_size = batch_size

        self.state_buffer = np.zeros((self.buffer_size,
                                      self.total_timestep+num_stack-1,
                                      self.num_states), dtype="float32")
        self.next_state_buffer = np.zeros((self.buffer_size,
                                           self.total_timestep+num_stack-1,
                                           self.num_states), dtype="float32")
        self.reward_buffer = np.zeros((self.buffer_size, self.total_timestep+num_stack-1), dtype="float32")
        self.action_buffer = np.zeros((self.buffer_size, self.total_timestep+num_stack-1, self.num_actions), dtype="float32")

        self.index = -1
        self.timestep_index = self.num_stack-1

    def reset(self):
        self.index += 1
        self.timestep_index = self.num_stack-1

    def append(self, state, reward, action, next_state):
        index = self.index % self.buffer_size

        self.state_buffer[index][self.timestep_index] = state
        self.reward_buffer[index][self.timestep_index] = reward
        self.action_buffer[index][self.timestep_index] = action
        self.next_state_buffer[index][self.timestep_index] = next_state

        self.timestep_index += 1

    def retrieve(self):
        record_range = min(self.index * self.total_timestep + self.timestep_index - self.num_stack + 1,
                           self.buffer_size * self.total_timestep)

        batch_indices = np.random.choice(record_range, self.batch_size)
        buffer_indices = batch_indices // self.total_timestep
        timestep_indices = batch_indices % self.total_timestep
        timestep_indices = np.linspace(timestep_indices, timestep_indices+self.num_stack-1, num=self.num_stack).astype(int)
        timestep_indices = np.transpose(timestep_indices)
        buffer_indices = np.repeat(buffer_indices, self.num_stack).reshape(self.batch_size, self.num_stack)

        state_batch = self.state_buffer[buffer_indices, timestep_indices]
        reward_batch = self.reward_buffer[buffer_indices, timestep_indices]
        action_batch = self.action_buffer[buffer_indices, timestep_indices]
        next_state_batch = self.next_state_buffer[buffer_indices, timestep_indices]

        return state_batch, reward_batch, action_batch, next_state_batch


if __name__ == "__main__":
    memory = Memory(10, 4, 2, 500, 32)
    memory.reset()
    memory.append([1, 2, 3, 4], 1, [1, 2], [1, 2, 3, 4])
    memory.append([1, 2, 3, 7], 2, [1, 2], [1, 2, 3, 4])
    memory.append([1, 2, 3, 6], 3, [1, 2], [1, 2, 3, 4])

    import time
    start = time.time()
    state, reward, action, next_state = memory.retrieve()
    print("{:.6f}".format(time.time()-start))
    print(state)

