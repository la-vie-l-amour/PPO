
MOUNTAIN_CAR_CONFIG = "SparseMountainCar-v0"
PENDULUM_CONFIG = "Pendulum-v1"
FROZENLAKE_CONFIG = "FrozenLake-v1"
VEHICLE_CONFIG = "Vehicle"
def print_configs():
    print(f"[{MOUNTAIN_CAR_CONFIG}, {PENDULUM_CONFIG} ,{FROZENLAKE_CONFIG}]")

def get_config(args):

    if args.config_name == PENDULUM_CONFIG:
        config = PendulumConfig()
    elif args.config_name == MOUNTAIN_CAR_CONFIG:
        config = MountainCarConfig()
    elif args.config_name == FROZENLAKE_CONFIG:
        config = FrozenLakeConfig()
    elif args.config_name == VEHICLE_CONFIG:
        config = VehicleConfig()
    else:
        raise ValueError("`{}` is not a valid config ID".format(args.config_name))

    # config.set_logdir(args.logdir)
    config.set_seed(args.seed)

    config.set_num_envs(args.num_envs)
    return config


class Config:
    def __init__(self):
        self.hidden_size = 32
        self.lr_actor = 1e-3
        self.lr_critic = 1e-2

        self.mini_batch_size = 256
        self.ppo_epochs = 30
        self.threshold_reward = -120
        self.num_steps = 128
        self.gamma = 0.99
        self.tau = 0.95
        self.clip_param = 0.2
        self.max_frames = 100000
        self.beta = 0.001
        self.num_envs = 16

        self.max_episode_len = 300

        self.seed = 10
    def set_seed(self, seed):
        self.seed = seed
    def set_num_envs(self, num_envs):
        self.num_envs = num_envs



class PendulumConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = "Pendulum-v1"
        self.threshold_reward = 1
        self.max_episode_len = 100
        self.max_frames = 100000

class VehicleConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = "Vehicle"
        self.max_episode_len = 100
        self.max_frames = 100000


class MountainCarConfig(Config):
    def __init__(self):
        super().__init__()
        self.env_name = "SparseMountainCar-v0"
        self.max_frames = 100000
        self.threshold_reward = 100




class FrozenLakeConfig(Config):
    
    def __init__(self):
        super().__init__()
        self.env_name = "FrozenLake-v1"
        self.max_frames = 100000







