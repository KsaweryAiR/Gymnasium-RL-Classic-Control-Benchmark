import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get_data(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.stack(self.values).squeeze()
        log_probs = torch.stack(self.log_probs).squeeze()
        dones = torch.tensor(self.dones, dtype=torch.float32)

        return states, actions, rewards, values, log_probs, dones