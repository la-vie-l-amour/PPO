import torch
import gym
import numpy as np


class Agent(object):
    def __init__(self, model, env):
        self.model = model
        self.env = env

    def run_eposide(self):
        with torch.no_grad():
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:

                state = np.expand_dims(state, axis=-1)

                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                dist, _ = self.model(state)
                next_state, reward, term, truncated, _ = env.step(dist.sample().cpu().numpy()[0])
                state = next_state
                if vis:
                    env.render()
                total_reward += reward
                done = term or truncated
        env.close()
        return total_reward

    def _add_action_noise(self):