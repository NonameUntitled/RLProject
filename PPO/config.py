import time
import random
import wandb
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from envs import make_env, VecPyTorch
from torch.utils.tensorboard import SummaryWriter


class Config:
    def __init__(self, env_id):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Running experiment with device: {}".format(self.device))

        self.env_id = env_id

        self.seed = 1
        self.num_env = 8

        self.env = VecPyTorch(
            DummyVecEnv([make_env(env_id, self.seed + i) for i in range(self.num_env)]),
            self.device
        )

        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

        self.update_every = 100
        self.num_learn = 4
        self.win_condition = 100
        self.n_steps = 7e6
        self.hidden_size = 64
        self.lr = 1e-5
        self.lr_annealing = False
        self.epsilon_annealing = False

        self.n_episodes = 2000
        self.max_t = 100

        self.epsilon = 0.1
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995

        self.gamma = 0.99

        self.memory = None
        self.gae = True
        self.gae_lambda = 0.95
        self.batch_size = 64
        self.buffer_size = int(1e5)

        self.learn_every = 4
        self.entropy_beta = 0.01

        self.model = None
        self.checkpoint_path = None
        self.save_iteration = 3
        self.save_dir = None

        self.wandb = False
        self.save_loc = None

        # Set up logging for tensor board
        experiment_name = f"{env_id}____{int(time.time())}"
        self.tb_logger = SummaryWriter(f"runs/{experiment_name}")

        self.init_seed()

    def init_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def init_wandb(self, project="rl-lib", entity="procgen"):
        wandb.init(project=project, entity=entity, sync_tensorboard=True)

        wandb.config.update({
            "seed": self.seed,
            "n_episodes": self.n_episodes,
            "max_t": self.max_t,
            "epsilon": self.epsilon,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay": self.eps_decay,
            "gamma": self.gamma,
            "lr": self.lr,
            "hidden_size": self.hidden_size,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "lr_annealing": self.lr_annealing,
            "learn_every": self.learn_every,
            "entropy_beta": self.entropy_beta,
            "update_every": self.update_every
        })

        self.wandb = True
