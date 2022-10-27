import torch
import functools

from stable_baselines3.common.vec_env import DummyVecEnv

from config import Config
from memory import Memory
from models import ActorCritic
from networks import CnnHeadModel, ActorModel, CriticModel
from envs import make_env, VecPyTorch
from dqn import DQNAgent

ENV_ID = "BreakoutNoFrameskip-v4"
CHECKPOINT_PATH = "./checkpoints_dqn/checkpoint_3072.pth"


def inference(config: Config):
    envs = VecPyTorch(
        DummyVecEnv([
            make_env(config.env_id, config.seed + i)
            for i in range(config.num_env)
        ]),
        config.device
    )

    agent = DQNAgent(config)

    states = envs.reset()

    while True:
        with torch.no_grad():
            actions, _, _, _ = agent.act(states)

        envs.render()

        next_states, rewards, dones, infos = envs.step(actions)

        states = next_states


def main():
    config = Config(ENV_ID)

    config.memory = Memory
    config.model = ActorCritic
    config.checkpoint_path = CHECKPOINT_PATH
    config.head_model = functools.partial(CnnHeadModel, config)
    config.actor_model = functools.partial(ActorModel, config)
    config.critic_model = functools.partial(CriticModel, config)

    inference(config)


if __name__ == "__main__":
    main()
