from envs import make_env,SubprocVecEnv, get_config

import argparse

def main(args):

    # create envs
    envs = [make_env(args.env_name,args.max_episode_len) for _ in range(args.num_envs)]
    envs = SubprocVecEnv(envs)
    num_actions = envs.action_space.shape[0]
    num_states = envs.observation_space.shape[0]

    # init



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--config_name", type=str,
                        default="SparseMountainCar-v0")  # Pendulum-v1, SparseMountainCar-v0, FrozenLake-v1
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()  # args, unknown = parser.parse_know_args()
    config = get_config(args)
    main(config)

