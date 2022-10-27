import os
import wandb
import torch
import functools
import numpy as np
from collections import deque
from stable_baselines3.common.vec_env import DummyVecEnv

from config import Config
from memory import Memory
from models import ActorCritic
from networks import CnnHeadModel, ActorModel, CriticModel
from envs import make_env, VecPyTorch
from ppo import PPOAgent

ENV_ID = "BreakoutNoFrameskip-v4"
SAVE_PATH = "./checkpoints_ppo"


def train(config: Config):
    envs = VecPyTorch(
        DummyVecEnv([
            make_env(config.env_id, config.seed + i)
            for i in range(config.num_env)
        ]),
        config.device
    )

    scores_deque = deque(maxlen=100)
    scores = []
    average_scores = []
    global_step = 0
    save_cnt = 0

    agent = PPOAgent(config)

    if config.wandb:
        wandb.watch(agent.model)

    while global_step < config.n_steps:
        states = envs.reset()

        values, dones = None, None

        while not agent.mem.is_full():
            global_step += config.num_env

            # Take actions
            with torch.no_grad():
                actions, log_probs, values, entrs = agent.act(states)

            next_states, rewards, dones, infos = envs.step(actions)

            # Add to memory buffer
            agent.add_to_mem(states, actions, rewards, log_probs, values, dones)

            # Update state
            states = next_states

            # Logging
            for info in infos:
                if 'episode' in info:
                    reward = info['episode']['r']

                    config.tb_logger.add_scalar("charts/episode_reward", reward, global_step)
                    if config.wandb:
                        wandb.log({
                            "episode_reward": reward,
                            "global_step": global_step
                        })

                    scores_deque.append(reward)
                    scores.append(reward)
                    average_scores.append(np.mean(scores_deque))

        # Parameters step
        value_loss, pg_loss, approx_kl, approx_entropy, lr_now = agent.learn(
            config.num_learn,
            values,
            dones,
            global_step
        )
        agent.mem.reset()

        save_cnt += 1
        if config.save_dir is not None and save_cnt % config.save_iteration == 0:
            torch.save(
                agent.model_old.state_dict(),
                os.path.join(config.save_dir, "checkpoint_{}.pth".format(global_step))
            )

        # Logging
        config.tb_logger.add_scalar("losses/value_loss", value_loss.item(), global_step)
        config.tb_logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        config.tb_logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        config.tb_logger.add_scalar("losses/approx_entropy", approx_entropy.item(), global_step)

        if config.wandb:
            wandb.log({
                "value_loss": value_loss,
                "policy_loss": pg_loss,
                "approx_kl": approx_kl,
                "approx_entropy": approx_entropy,
                "global_step": global_step,
                "learning_rate": lr_now
            })

        print("Global Step: {}  Average Score: {:.2f}".format(global_step, np.mean(scores_deque)))

    return scores, average_scores


def main():
    config = Config(ENV_ID)

    config.update_every = 128
    config.num_learn = 4
    config.win_condition = 230
    config.n_steps = 7e6
    config.hidden_size = 512
    config.lr = 3e-4
    config.lr_annealing = True
    config.epsilon_annealing = True

    config.memory = Memory
    config.model = ActorCritic
    config.save_dir = SAVE_PATH
    config.head_model = functools.partial(CnnHeadModel, config)
    config.actor_model = functools.partial(ActorModel, config)
    config.critic_model = functools.partial(CriticModel, config)

    # config.init_wandb()

    final_score, _ = train(config)

    with open('final_score_ppo.txt', 'w') as f:
        f.write(' '.join(list(map(str, final_score))))

    print("Final global score: {}".format(final_score))


if __name__ == "__main__":
    main()
