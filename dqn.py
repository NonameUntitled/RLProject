import torch
import torch.nn as nn
import torch.optim as optim


class DQNAgent:

    def __init__(self, config):
        self.mem = config.memory(
            config.update_every,
            config.num_env,
            config.env,
            config.device,
            config.gamma,
            config.gae_lambda
        )

        self.lr = config.lr
        self.n_steps = config.n_steps
        self.lr_annealing = config.lr_annealing
        self.device = config.device

        self.model = config.model(config).to(self.device)

        if config.checkpoint_path is not None:
            self.model.load_state_dict(torch.load(config.checkpoint_path, map_location=torch.device('cpu')))
            print("Model was loaded from checkpoint!")

        self.model_old = config.model(config).to(self.device)

        self.model_old.load_state_dict(self.model.state_dict())

        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)

        self.config = config

    def add_to_mem(self, state, action, reward, log_prob, values, done):
        self.mem.add(state, action, reward, log_prob, values, done)

    def act(self, x):
        x = x.to(self.config.device)
        return self.model_old.act(x)

    def learn(self, num_learn, last_value, next_done, global_step):
        frac = 1.0 - (global_step - 1.0) / self.n_steps
        lr_now = self.lr * frac

        if self.lr_annealing:
            self.optimiser.param_groups[0]['lr'] = lr_now

        self.mem.calculate_advantage_gae(last_value, next_done)

        for i in range(num_learn):
            for mini_batch_idx in self.mem.get_mini_batch_idxs(mini_batch_size=256):
                prev_states, prev_actions, prev_log_probs, discounted_returns, advantage, prev_values = self.mem.sample(
                    mini_batch_idx
                )

                # Calculate losses
                new_values = self.model.get_values(prev_states).view(-1)

                loss = ((new_values - discounted_returns) ** 2).sum()

                # calculate gradient
                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimiser.step()

        return loss, None, None, None, lr_now
