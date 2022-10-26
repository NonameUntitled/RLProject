import torch.nn as nn
from torch.distributions import Categorical

from networks import layer_init


class Scale(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class ActorCritic(nn.Module):

    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.network = nn.Sequential(
            Scale(1 / 255),
            layer_init(nn.Conv2d(4, 32, (8, 8), stride=(4, 4))),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, (4, 4), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, (3, 3), stride=(1, 1))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.ReLU()
        )  # Shared network

        self.actor = layer_init(nn.Linear(512, config.action_space), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        return self.network(x)

    def act(self, x, action=None):
        values = self.critic(self.forward(x))
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), values, probs.entropy()

    def get_values(self, x):
        return self.critic(self.forward(x))
