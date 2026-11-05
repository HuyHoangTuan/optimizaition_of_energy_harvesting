"""Synchronous collection with independent local-reward learning."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import uuid

import numpy as np
import torch

from . import MODEL_VERSION, SCHEMA_VERSION
from .agent import AgentConfig, IndependentDQN
from .environment import EnvironmentConfig, JointEnvironment


class MultiAgentTrainer:
    def __init__(self, environment_config=EnvironmentConfig(), agent_config=AgentConfig(), *, trace_bank=None, actions=None):
        self.environment_config, self.agent_config = environment_config, agent_config
        self.environment = JointEnvironment(environment_config, trace_bank=trace_bank, actions=actions)
        self.agents = [IndependentDQN(i, self.environment.observation_size, len(self.environment.actions),
                                     environment_config.seed + 1009 * (i + 1), agent_config)
                       for i in range(environment_config.num_su)]

    def train_episode(self):
        observations = self.environment.reset()
        records = []
        terminal = False
        while not terminal:
            actions = [agent.select_action(observations[i]) for i, agent in enumerate(self.agents)]
            following, rewards, terminal, info = self.environment.step(actions)
            losses = []
            for i, agent in enumerate(self.agents):
                agent.remember(observations[i], actions[i], rewards[i], following[i], terminal)
                losses.append(agent.optimize())
            records.append({**info, "observations": observations.tolist(), "next_observations": following.tolist(),
                            "terminal": terminal, "losses": losses})
            observations = following
        for agent in self.agents:
            agent.end_episode()
        return records

    def run(self, episodes=1, output=None):
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
            raise ValueError("episodes must be a positive integer")
        if output is None:
            output = Path(".cache/multi_agent") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10])
        output = Path(output)
        # Atomic reservation; even an existing empty output directory is refused.
        output.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        package = Path(__file__).parent
        project = package.parents[1]
        sources = list(package.glob("*.py")) + [project / "train_multi_agent.py", project / "requirements-multi-agent.txt"]
        metadata = {"schema_version": SCHEMA_VERSION, "model_version": self.environment.model_version,
                    "environment_config": asdict(self.environment_config), "agent_config": asdict(self.agent_config),
                    "episodes": episodes, "reward_scope": "own_su", "device": "cpu", "optimizer": self.agent_config.optimizer,
                    "seed": self.environment_config.seed, "agent_seeds": [a.seed for a in self.agents],
                    "action_map": self.environment.actions,
                    "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
                    "platform": platform.platform(), "started_utc": datetime.now(timezone.utc).isoformat(),
                    "source_sha256": {str(p.relative_to(project)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    "checkpoint_contract": "one independent policy/target/optimizer per SU; no exact resume"}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        totals = np.zeros(len(self.agents))
        with (output / "transitions.jsonl").open("x", encoding="utf-8") as stream:
            for episode in range(episodes):
                for record in self.train_episode():
                    totals += record["rewards"]
                    stream.write(json.dumps({"episode": episode, **record}, allow_nan=False) + "\n")
        for agent in self.agents:
            checkpoint = {**agent.checkpoint(), "model_version": self.environment.model_version,
                          "action_map": self.environment.actions, "config": asdict(self.agent_config)}
            torch.save(checkpoint, output / ("su_%03d.pt" % agent.su_id))
        summary = {"episodes": episodes, "slots": episodes * self.environment_config.slots,
                   "per_su_reward": totals.tolist(), "network_reward_diagnostic": float(totals.sum()),
                   "updates": [a.updates for a in self.agents],
                   "runtime_seconds": time.perf_counter() - started}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        return output, summary
