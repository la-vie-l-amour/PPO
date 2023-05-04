import torch
import gym
import numpy as np


class Agent(object):
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cuda:0')

    def run_eposide(self):
        state = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            state = np.expand_dims(state, axis=-1)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _ = self.model(state)
            action = dist.sample().cpu().numpy()[0]
            # 是否添加noise
            action = self._add_action_noise(action, noise = 0.01)
            next_state, reward, term, truncated, _ = self.env.step(action)
            state = next_state
            total_reward += reward
            done = term or truncated
        self.env.close()
        return total_reward

    def _add_action_noise(self,action,noise):
        if noise is not None:
            action = action + noise * torch.randn_like(action)
        return action
