"""Simultaneous joint-action compatibility score environment (not validated physics)."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np

from . import MODEL_VERSION


@dataclass(frozen=True)
class EnvironmentConfig:
    num_su: int = 2
    num_pu: int = 2
    slots: int = 20
    seed: int = 2
    capacity: float = 0.5
    slot_duration: float = 1.0
    efficiency: float = 0.9
    ambient_max: float = 0.2
    noise: float = 1.0
    penalty: float = 1.0
    primary_threshold: float = 0.5
    primary_switch: int = 10
    dynamic_rho: bool = False
    debit_mode: str = "executed"

    def __post_init__(self):
        if self.debit_mode not in ("executed", "requested"):
            raise ValueError("debit_mode must be executed or requested")
        for name in ("num_su", "num_pu", "slots"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if self.num_pu != 2:
            raise ValueError("legacy_score_joint_actions_v1 requires exactly two PUs")
        for name in ("capacity", "slot_duration", "ambient_max", "noise", "primary_threshold"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.ambient_max < 0.001 or not 0 <= self.efficiency <= 1:
            raise ValueError("ambient_max >= .001 and efficiency in [0,1] required")
        if not np.isfinite(self.penalty) or self.penalty < 0:
            raise ValueError("penalty must be finite and nonnegative")
        if not isinstance(self.primary_switch, Integral) or self.primary_switch < 0:
            raise ValueError("primary_switch must be a nonnegative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def action_map(dynamic_rho=False):
    """Stable, unshuffled (k, watts, rho) map; harvest retains legacy P/rho branches."""
    rhos = [i / 8 for i in range(1, 8)] if dynamic_rho else [0.4]
    return tuple((k, p / 16, rho) for rho in rhos for p in range(1, 17) for k in (0, 1))


def dbm(power):
    return 10 * np.log10(np.asarray(power) * 1000)


class JointEnvironment:
    model_version = MODEL_VERSION

    def __init__(self, config=EnvironmentConfig(), trace=None, actions=None, trace_bank=None):
        self.config = config
        self.actions = tuple(actions) if actions is not None else action_map(config.dynamic_rho)
        if not self.actions:
            raise ValueError("nonempty action map required")
        for k, p, rho in self.actions:
            # Scores intentionally use numeric dBm: positive powers >= .001 prevent negative terms.
            if k not in (0, 1) or not np.isfinite(p) or p < .001 or not np.isfinite(rho) or not 0 <= rho < 1:
                raise ValueError("actions require k in {0,1}, P >= .001, 0 <= rho < 1")
        self.rng = np.random.default_rng(config.seed)
        self.fixed_trace = self._validate_trace(trace) if trace is not None else None
        if trace is not None and trace_bank is not None:
            raise ValueError("supply trace or trace_bank, not both")
        self.trace_bank = tuple(self._validate_trace(t) for t in trace_bank) if trace_bank is not None else None
        if self.trace_bank is not None and not self.trace_bank:
            raise ValueError("trace_bank must be nonempty")
        self.episode_index = 0
        self.model_version = MODEL_VERSION + ("_requested_debit" if config.debit_mode == "requested" else "")
        self.trace = None
        self.time = 0
        self.records = []

    @property
    def observation_size(self):
        return 3 + self.config.num_su + 3 * self.config.num_pu

    def _validate_trace(self, trace):
        c = self.config
        shapes = {"rt": (c.slots, c.num_su, c.num_su),
                  "sp": (c.slots, c.num_su, c.num_pu),
                  "pr": (c.slots, c.num_su, c.num_pu),
                  "ps": (c.slots, c.num_su, c.num_pu),
                  "primary_power": (c.slots, c.num_su, c.num_pu),
                  "ambient": (c.slots,)}
        result = {}
        for key, shape in shapes.items():
            a = np.array(trace[key], dtype=np.float64, copy=True)
            if a.shape != shape or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError("invalid trace field " + key)
            if key == "primary_power" and (a < .001).any():
                raise ValueError("primary_power must be >= .001 for numeric dBm scores")
            a.flags.writeable = False
            result[key] = a
        return result

    def _density_grid(self, count):
        """Legacy RandomUtils.rayleigh density-grid construction with Lambda=1."""
        exp = self.rng.exponential(1.0, self.config.slots)
        rate = min(1 / np.mean(exp), 1.0)
        return rate * np.exp(-rate * np.linspace(exp.min(), exp.max(), count))

    def _generate_trace(self):
        c = self.config
        result = {}
        for key, shape in (("rt", (c.num_su, c.num_su)),
                           ("sp", (c.num_su, c.num_pu)),
                           ("pr", (c.num_su, c.num_pu)),
                           ("ps", (c.num_su, c.num_pu))):
            a = np.empty((c.slots,) + shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    a[:, i, j] = self._density_grid(c.slots)
            result[key] = a
        result["primary_power"] = self.rng.uniform(.001, 1, (c.slots, c.num_su, c.num_pu))
        result["ambient"] = self.rng.uniform(.001, c.ambient_max, c.slots)
        return self._validate_trace(result)

    def reset(self, episode_index=None):
        index = self.episode_index if episode_index is None else episode_index
        if self.trace_bank is not None:
            if not 0 <= index < len(self.trace_bank):
                raise IndexError("trace bank exhausted")
            self.trace = self.trace_bank[index]
        else:
            self.trace = self.fixed_trace if self.fixed_trace is not None else self._generate_trace()
        self.episode_index = index + 1
        self.time = 0
        n = self.config.num_su
        self.battery = np.zeros(n)
        self.harvest = np.zeros(n)
        self.debit = np.zeros(n)
        self.records = []
        return np.zeros((n, self.observation_size), dtype=np.float64)

    def step(self, action_indices):
        c = self.config
        # Validate the complete vector before time, RNG, records or battery mutation.
        try:
            indices = list(action_indices)
        except TypeError as exc:
            raise ValueError("one action index per SU required") from exc
        if len(indices) != c.num_su or any(isinstance(a, (bool, np.bool_)) or not isinstance(a, Integral)
                                          or not 0 <= a < len(self.actions) for a in indices):
            raise ValueError("invalid joint action vector")
        if self.trace is None:
            raise RuntimeError("reset before step")
        if self.time >= c.slots:
            raise RuntimeError("episode already terminal; reset required")
        indices = [int(a) for a in indices]
        t = self.time
        v = 1 if t + 1 <= c.primary_switch else 0
        k, p, rho = np.asarray([self.actions[a] for a in indices]).T
        mu = 1 - rho
        battery = np.clip(self.battery + self.harvest - self.debit, 0, c.capacity)
        rt, sp, pr, ps = (self.trace[key][t] for key in ("rt", "sp", "pr", "ps"))
        primary = self.trace["primary_power"][t, :, v]
        harvest = rho * c.slot_duration * primary * c.efficiency * ps[:, v] + self.trace["ambient"][t]
        costs = mu * p * c.slot_duration
        executed = (k == 0) & (costs <= battery) & (mu * p * sp[:, v] <= c.primary_threshold)
        debit = np.where((k == 0) if c.debit_mode == "requested" else executed, costs, 0)
        weighted = executed * mu * dbm(p)
        # Exclude self explicitly to avoid cancellation noise in the one-SU case.
        cross = rt.copy()
        np.fill_diagonal(cross, 0)
        interference = cross @ weighted
        denominator = c.noise + interference
        if v == 1:
            denominator = denominator + dbm(primary) * pr[:, v]
        scores = c.slot_duration * np.log2(1 + mu * dbm(p) * np.diag(rt) / denominator)
        rewards = np.full(c.num_su, -c.penalty, dtype=np.float64)
        rewards[(k == 1) & (costs > battery)] = 0
        rewards[executed] = scores[executed]
        observation = np.concatenate((np.full((c.num_su, 1), v), self.harvest[:, None],
                                      battery[:, None], rt, sp, pr, ps), axis=1)
        record = {"slot": t + 1, "primary": v, "action_indices": indices,
                  "actions": np.asarray([self.actions[a] for a in indices]).tolist(),
                  "rewards": rewards.tolist(), "executed": executed.tolist(),
                  "scores": np.where(executed, scores, 0).tolist(),
                  "attempted_transmit": (k == 0).tolist(),
                  "battery_constraint_failed": ((k == 0) & (costs > battery)).tolist(),
                  "primary_constraint_failed": ((k == 0) & (mu * p * sp[:, v] > c.primary_threshold)).tolist(),
                  "battery": battery.tolist(), "harvest": harvest.tolist(),
                  "debit": debit.tolist(),
                  "interference": interference.tolist(),
                  "network_reward": float(rewards.sum()),
                  "network_score": float(scores[executed].sum())}
        self.battery, self.harvest = battery, harvest
        self.debit = debit
        self.records.append(record)
        self.time += 1
        return observation, rewards, self.time == c.slots, record
