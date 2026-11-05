"""A separate feed-forward DQN, target, optimizer and replay per SU."""
from collections import deque
from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class AgentConfig:
    hidden: int = 64
    batch_size: int = 32
    replay_capacity: int = 10000
    learning_rate: float = .001
    gamma: float = .99
    epsilon_start: float = 1.0
    epsilon_end: float = .05
    epsilon_decay: float = 500.0
    target_interval: int = 100
    optimizer: str = "Adam"
    loss: str = "huber"
    gradient_clip: float | None = 10.0
    exploration: str = "steps"
    forced_random_episodes: int = 500
    source_epsilon_increment: float = 1.5
    target_schedule: str = "updates"
    target_tau: float = .001
    target_episode_interval: int = 5
    double_dqn: bool = False

    @classmethod
    def legacy_parity(cls, **overrides):
        values = dict(hidden=128, batch_size=64, replay_capacity=100000,
                      learning_rate=.003, epsilon_end=.01, epsilon_decay=.001,
                      optimizer="SGD", loss="mse", gradient_clip=None,
                      exploration="source", target_schedule="episodes")
        return cls(**{**values, **overrides})

    def __post_init__(self):
        for name in ("hidden", "batch_size", "replay_capacity", "target_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("invalid gamma or epsilon range")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or not math.isfinite(self.epsilon_decay) or self.epsilon_decay <= 0:
            raise ValueError("learning_rate and epsilon_decay must be finite and positive")
        if self.optimizer not in ("Adam", "SGD") or self.loss not in ("huber", "mse"):
            raise ValueError("unsupported optimizer or loss")
        if self.exploration not in ("steps", "source") or self.target_schedule not in ("updates", "episodes"):
            raise ValueError("unsupported exploration or target schedule")
        if self.gradient_clip is not None and (not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0):
            raise ValueError("gradient_clip must be positive or None")
        if not 0 < self.target_tau <= 1 or not isinstance(self.target_episode_interval, int) or self.target_episode_interval < 1:
            raise ValueError("invalid episode target schedule")
        if not isinstance(self.forced_random_episodes, int) or self.forced_random_episodes < 0 or not math.isfinite(self.source_epsilon_increment) or self.source_epsilon_increment <= 0:
            raise ValueError("invalid source exploration schedule")


class IndependentDQN:
    def __init__(self, su_id, observation_size, num_actions, seed, config=AgentConfig()):
        self.su_id, self.seed, self.config = su_id, seed, config
        self.observation_size, self.num_actions = observation_size, num_actions
        self.rng = random.Random(seed)
        self.exploration_rng = np.random.default_rng(seed)
        # Initialization has its own deterministic seed without changing caller torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.policy = nn.Sequential(nn.Linear(observation_size, config.hidden), nn.ReLU(),
                                        nn.Linear(config.hidden, config.hidden), nn.ReLU(),
                                        nn.Linear(config.hidden, num_actions))
            self.target = nn.Sequential(nn.Linear(observation_size, config.hidden), nn.ReLU(),
                                        nn.Linear(config.hidden, config.hidden), nn.ReLU(),
                                        nn.Linear(config.hidden, num_actions))
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.target.requires_grad_(False)
        self.optimizer = getattr(torch.optim, config.optimizer)(self.policy.parameters(), lr=config.learning_rate)
        self.replay = deque(maxlen=config.replay_capacity)
        self.selections = 0
        self.updates = 0
        self.episodes = 0
        self.epsilon_drop = 0.0
        self.last_diagnostics = None

    def select_action(self, observation, episode_index=None):
        c = self.config
        episode_index = self.episodes if episode_index is None else episode_index
        if c.exploration == "source":
            sample = self.exploration_rng.random()
            epsilon = c.epsilon_end + (c.epsilon_start - c.epsilon_end) * math.exp(-self.epsilon_drop * c.epsilon_decay)
            if episode_index < c.forced_random_episodes:
                sample = -1
                self.epsilon_drop = 0.0
            else:
                self.epsilon_drop += c.source_epsilon_increment
            random_action = sample <= epsilon
        else:
            epsilon = c.epsilon_end + (c.epsilon_start - c.epsilon_end) * math.exp(-self.selections / c.epsilon_decay)
            random_action = self.rng.random() < epsilon
        self.selections += 1
        if random_action:
            return self.rng.sample(range(self.num_actions), 1)[0] if c.exploration == "source" else self.rng.randrange(self.num_actions)
        with torch.no_grad():
            return int(self.policy(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)).argmax().item())

    def end_episode(self, episode_index=None):
        index = self.episodes if episode_index is None else episode_index
        if self.config.target_schedule == "episodes" and index % self.config.target_episode_interval == 0:
            tau = self.config.target_tau
            self.target.load_state_dict({k: v * tau + self.target.state_dict()[k] * (1-tau)
                                         for k, v in self.policy.state_dict().items()})
        self.episodes = index + 1

    def remember(self, observation, action, reward, next_observation, terminal):
        self.replay.append((np.array(observation, dtype=np.float32, copy=True), int(action), float(reward),
                            np.array(next_observation, dtype=np.float32, copy=True), bool(terminal)))

    def optimize(self):
        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.rng.sample(list(self.replay), self.config.batch_size)
        return self.optimize_batch(batch)

    def optimize_batch(self, batch):
        """Apply one explicit replay batch; also supports terminal-only batches."""
        states, actions, rewards, next_states, terminals = zip(*batch)
        states = torch.from_numpy(np.stack(states))
        next_states = torch.from_numpy(np.stack(next_states))
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        nonterminal = ~torch.tensor(terminals, dtype=torch.bool)
        predicted = self.policy(states).gather(1, actions)
        # Also valid for a terminal-only batch; never concatenate an empty list.
        next_values = torch.zeros(len(batch))
        with torch.no_grad():
            if nonterminal.any():
                target_values = self.target(next_states[nonterminal])
                if self.config.double_dqn:
                    selected = self.policy(next_states[nonterminal]).argmax(1, keepdim=True)
                    next_values[nonterminal] = target_values.gather(1, selected).squeeze(1)
                else:
                    next_values[nonterminal] = target_values.max(1).values
            expected = rewards + self.config.gamma * next_values
        criterion = nn.functional.mse_loss if self.config.loss == "mse" else nn.functional.smooth_l1_loss
        loss = criterion(predicted, expected.unsqueeze(1))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad.detach() for p in self.policy.parameters() if p.grad is not None]
        grad_norm = sum(float(g.square().sum()) for g in gradients) ** .5
        if self.config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.gradient_clip)
        self.last_diagnostics = {"loss": float(loss), "gradient_norm": grad_norm,
                                 "selected_q_mean": float(predicted.detach().mean()),
                                 "target_mean": float(expected.mean())}
        self.optimizer.step()
        self.updates += 1
        if self.config.target_schedule == "updates" and self.updates % self.config.target_interval == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def checkpoint(self):
        return {"su_id": self.su_id, "seed": self.seed, "observation_size": self.observation_size,
                "num_actions": self.num_actions, "policy": self.policy.state_dict(),
                "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                "selections": self.selections, "updates": self.updates,
                "purpose": "policy inspection/evaluation; not an exact-resume snapshot"}
