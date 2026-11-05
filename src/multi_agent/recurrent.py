"""Independent own-reward DRQN with complete chronological episode replay."""
from collections import deque
import math
import random

import numpy as np
import torch
from torch import nn

from .agent import AgentConfig, IndependentDQN


class RecurrentQNetwork(nn.Module):
    def __init__(self, observation_size, num_actions, hidden=64):
        super().__init__()
        self.gru = nn.GRU(observation_size, hidden, num_layers=1, batch_first=True,
                          dropout=0, bidirectional=False)
        self.head = nn.Linear(hidden, num_actions)

    def forward(self, observations, hidden=None):
        if observations.ndim != 3:
            raise ValueError("observations must have shape [batch,time,observation]")
        values, hidden = self.gru(observations, hidden)
        return self.head(values), hidden


class IndependentDRQN(IndependentDQN):
    """Fixed protocol learner; checkpoints support evaluation, not exact resume."""
    episode_slots = 20
    episodes_per_update = 4
    loss_positions = 64

    def __init__(self, su_id, observation_size, num_actions, seed,
                 config=None):
        config = config or AgentConfig.legacy_parity(hidden=64)
        if config != AgentConfig.legacy_parity(hidden=64):
            raise ValueError("DRQN requires the frozen legacy-parity GRU64 settings")
        self.su_id, self.seed, self.config = su_id, seed, config
        self.observation_size, self.num_actions = observation_size, num_actions
        self.rng = random.Random(seed)
        self.mask_rng = random.Random(seed + 7919)
        self.exploration_rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.policy = RecurrentQNetwork(observation_size, num_actions)
            self.target = RecurrentQNetwork(observation_size, num_actions)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=config.learning_rate)
        self.replay = deque(maxlen=5000)
        self.pending = []
        self.acting_prefix = []
        self.hidden = None
        self.selections = self.updates = self.episodes = 0
        self.epsilon_drop = 0.0
        self.last_diagnostics = None

    def _observation(self, value):
        result = np.array(value, dtype=np.float32, copy=True)
        if result.shape != (self.observation_size,) or not np.isfinite(result).all():
            raise ValueError("invalid local observation")
        return result

    def select_action(self, observation, episode_index=None):
        observation = self._observation(observation)
        # Every decision processes the current observation, including random decisions.
        with torch.no_grad():
            q, hidden = self.policy(torch.from_numpy(observation)[None, None], self.hidden)
        self.hidden = hidden.detach()
        self.acting_prefix.append(observation)
        c = self.config
        index = self.episodes if episode_index is None else episode_index
        sample = self.exploration_rng.random()
        epsilon = c.epsilon_end + (c.epsilon_start-c.epsilon_end) * math.exp(-self.epsilon_drop*c.epsilon_decay)
        if index < c.forced_random_episodes:
            sample = -1
            self.epsilon_drop = 0.0
        else:
            self.epsilon_drop += c.source_epsilon_increment
        self.selections += 1
        return self.rng.sample(range(self.num_actions), 1)[0] if sample <= epsilon else int(q[0, -1].argmax())

    def remember(self, observation, action, reward, next_observation, terminal):
        observation, following = self._observation(observation), self._observation(next_observation)
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= action < self.num_actions:
            raise ValueError("invalid local action")
        if not np.isfinite(reward) or not isinstance(terminal, (bool, np.bool_)):
            raise ValueError("invalid reward or terminal")
        if self.pending and not np.array_equal(self.pending[-1][3], observation):
            raise ValueError("adjacent episode observations do not match")
        position = len(self.pending) + 1
        if position > self.episode_slots or bool(terminal) != (position == self.episode_slots):
            raise ValueError("only complete 20-slot episodes with final terminal are accepted")
        self.pending.append((observation, int(action), float(reward), following, bool(terminal)))
        if terminal:
            observations = np.stack([row[0] for row in self.pending] + [following])
            episode = dict(observations=observations,
                           actions=np.array([row[1] for row in self.pending], dtype=np.int64),
                           rewards=np.array([row[2] for row in self.pending], dtype=np.float32),
                           terminals=np.array([row[4] for row in self.pending], dtype=bool))
            self.replay.append(episode)
            self.pending.clear()

    def td_values(self, episodes):
        """Expose aligned values for semantic verification; no learner mutation."""
        observations = torch.from_numpy(np.stack([e['observations'] for e in episodes]))
        actions = torch.from_numpy(np.stack([e['actions'] for e in episodes]))
        rewards = torch.from_numpy(np.stack([e['rewards'] for e in episodes]))
        terminals = torch.from_numpy(np.stack([e['terminals'] for e in episodes]))
        online, _ = self.policy(observations)
        predicted = online[:, :-1].gather(2, actions[..., None]).squeeze(2)
        with torch.no_grad():
            target, _ = self.target(observations)
            expected = rewards + self.config.gamma * (~terminals) * target[:, 1:].max(2).values
        return predicted, expected

    def optimize(self):
        if len(self.replay) < self.episodes_per_update:
            return None
        episodes = self.rng.sample(list(self.replay), self.episodes_per_update)
        predicted, expected = self.td_values(episodes)
        positions = self.mask_rng.sample(range(80), self.loss_positions)
        indices = torch.tensor(positions, dtype=torch.long)
        loss = nn.functional.mse_loss(predicted.reshape(-1)[indices], expected.reshape(-1)[indices])
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad.detach() for p in self.policy.parameters() if p.grad is not None]
        grad_norm = sum(float(g.square().sum()) for g in gradients) ** .5
        self.optimizer.step()
        self.updates += 1
        self.last_diagnostics = dict(loss=float(loss.detach()), gradient_norm=grad_norm,
            selected_q_mean=float(predicted.detach().reshape(-1)[indices].mean()),
            target_mean=float(expected.reshape(-1)[indices].mean()),
            loss_positions=positions, loss_count=64, processed_transitions=80,
            processed_observations_per_network=84)
        # Rebuild the processed acting prefix with the new parameters before acting again.
        self.refresh_hidden()
        return float(loss.detach())

    def refresh_hidden(self):
        self.hidden = None
        if self.acting_prefix:
            with torch.no_grad():
                _, hidden = self.policy(torch.from_numpy(np.stack(self.acting_prefix))[None])
            self.hidden = hidden.detach()

    def reset_hidden(self):
        self.hidden = None
        self.acting_prefix.clear()

    def end_episode(self, episode_index=None):
        if self.pending:
            raise ValueError("incomplete episode cannot be ended")
        super().end_episode(episode_index)
        self.reset_hidden()

    def checkpoint(self):
        return {**super().checkpoint(), "learner": "IndependentDRQN",
                "config": self.config.__dict__, "episode_capacity": 5000,
                "replay_episodes": len(self.replay), "episodes": self.episodes,
                "purpose": "evaluation only; excludes replay, acting state and RNG; not exact resume"}
