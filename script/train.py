import pickle

from envs import make_env,SubprocVecEnv, get_config, Env

import argparse
from model import ActorCritic
import torch
import numpy as np
import torch.optim as optim
import os
from IPython.display import clear_output
import pylab

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def path_isExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def plot(frame_idx, rewards,test_stds, env_name):
    clear_output(True)
    fig = pylab.figure(figsize =(12,10))
    pylab.plot(rewards, 'b')
    pylab.xlabel('iteration')
    pylab.ylabel('reward')
    pylab.title(f'PPO {np.mean(np.array(rewards))}')
    path_isExist(f"./graphs/{env_name}")
    pylab.savefig(f"./graphs/{env_name}/{env_name}.png", bbox_inches = "tight")
    pylab.close(fig)
    # 保存rewards
    with open("./graphs/rewads_buffer", 'wb') as f:
        pickle.dump(rewards, f)
    with open("./graphs/stds_buffer", 'wb') as f:
        pickle.dump(test_stds, f)

def eval(model, env_name, max_episode_len, vis=False):
    env = Env(env_name, max_episode_len)
    with torch.no_grad():
        state = env.reset()
        if vis:
            env.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = model(state)
            next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            if vis:
                env.render()
            total_reward += reward
    env.close()
    return total_reward

def main(args):
    # create envs
    envs = [make_env(args.env_name,args.max_episode_len) for _ in range(args.num_envs)]
    envs = SubprocVecEnv(envs)

    num_actions = envs.action_space.shape[0]
    num_states = envs.observation_space.shape[0]

    model = ActorCritic(num_states, num_actions, args.hidden_size).to(device)
    optimizer = optim.Adam([
        {'params': model.actor.parameters(), 'lr': args.lr_actor},
        {'params': model.critic.parameters(), 'lr': args.lr_critic}
    ])

    frame_idx = 0
    test_rewards = []
    test_stds = []
    state = envs.reset()

    early_stop = False

    # log_probs = []
    # values = []
    # states = []
    # actions = []
    # rewards = []
    # masks = []

    while frame_idx < args.max_frames and not early_stop:
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0
        # global next_state
        global next_state
        # collect data
        for _ in range(args.num_steps):

            state = torch.FloatTensor(state).to(device)

            dist, value = model(state)
            action = dist.sample()

            next_state, reward, done,_ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)

            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)

            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            states.append(state)
            actions.append(action)

            state = next_state
            frame_idx += 1

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values, args.gamma, args.tau)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantages = returns - values

        # PPO update
        for _ in range(args.ppo_epochs):
            for state_, action_, old_log_probs, return_, advantage in ppo_iter(args.mini_batch_size, states, actions,
                                                                               log_probs, returns, advantages):
                dist, value = model(state_)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action_)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()  # critic_loss 关于这个

                loss = 0.5 * critic_loss + actor_loss - args.beta * entropy  # 这里将entropy放入loss，是为了增大它的熵，也即扩大explore

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards_list = [eval(model, args.env_name, args.max_episode_len) for _ in range(10)]
        test_reward = np.mean(rewards_list)  # 这里是否可以传参数model
        test_std = np.std(rewards_list)
        test_rewards.append(test_reward)
        test_stds.append(test_std)

        print('++++', test_reward, '+++++', frame_idx, "++++")
        plot(frame_idx, test_rewards,test_stds, args.env_name)

  # 产生mini_batch_size数据用于更新参数
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.split(ids, batch_size // mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :]

    # L = states.size(0)
    # ids = np.random.permutation(L)
    # for i in range(0, L, mini_batch_size):
    #     j = min(L, i + mini_batch_size)
    #     batch_ids = ids[i:j]
    #     yield states[batch_ids, :], actions[batch_ids, :], log_probs[batch_ids, :], returns[batch_ids, :], advantage[batch_ids, :]



# Generalized Advantage Estimator 一种用于估计Advantage的方法
def compute_gae(next_value, rewards, masks, values, gamma, tau):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--config_name", type=str,
                        default="Vehicle")  # Pendulum-v1, SparseMountainCar-v0, FrozenLake-v1
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()  # args, unknown = parser.parse_know_args()
    config = get_config(args)
    main(config)

