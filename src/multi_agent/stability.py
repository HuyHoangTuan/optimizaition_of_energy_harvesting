"""Frozen evaluation and deterministic selection for controlled learner studies."""
import hashlib
import math
import random

import numpy as np
import torch

from .environment import JointEnvironment, dbm


def fingerprint(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.detach().cpu().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key, part in item.items():
                visit(key)
                visit(part)
        elif isinstance(item, (tuple, list)) or type(item).__name__ == "deque":
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def training_fingerprint(agents):
    return fingerprint({"agents": [{"policy": a.policy.state_dict(), "target": a.target.state_dict(),
        "optimizer": a.optimizer.state_dict(), "replay": a.replay,
        "counters": (a.selections, a.updates, a.episodes, a.epsilon_drop),
        "diagnostics": a.last_diagnostics, "python_rng": a.rng.getstate(),
        "gradients": [p.grad for p in a.policy.parameters()],
        "requires_grad": [p.requires_grad for p in a.policy.parameters()],
        "numpy_rng": a.exploration_rng.bit_generator.state,
        "modes": (a.policy.training, a.target.training)} for a in agents],
        "global_python": random.getstate(), "global_numpy": np.random.get_state(),
        "global_torch": torch.random.get_rng_state()})


def q_bounds(config, actions, bank, gamma):
    """Conservative continuing-return bound, only for nonnegative dBm interference."""
    if gamma >= 1:
        raise ValueError("finite bound requires gamma < 1")
    if any(np.max(t["rt"]) > 1 or np.min(t["rt"]) < 0 for t in bank):
        raise ValueError("bound assumes channel density gains in [0,1]")
    if any(p < .001 for _, p, _ in actions):
        raise ValueError("bound assumes nonnegative numeric dBm power")
    maximum = max(config.slot_duration * math.log2(1 + (1-rho)*float(dbm(p))/config.noise)
                  for _, p, rho in actions)
    return [-config.penalty/(1-gamma), maximum/(1-gamma)]


def evaluate(agents, config, actions, bank, seed, *, policies=None, random_policy=False, shared=False, random_agent=None, retain_slots=False):
    """Never select through learner methods; verify every populated training state."""
    before = training_fingerprint(agents)
    policies = [a.policy for a in agents] if policies is None else policies
    env = JointEnvironment(config, actions=actions, trace_bank=bank)
    rngs = [random.Random(seed + i*1009) for i in range(config.num_su)]
    rows, selected_q, slots_retained = [], [[] for _ in range(config.num_su)], []
    batteries = []
    with torch.no_grad():
        for episode in range(len(bank)):
            observation = env.reset()
            totals = {key: np.zeros(config.num_su) for key in (
                "reward", "score", "executed", "attempted", "battery_failed", "primary_failed",
                "debit", "battery", "harvest", "power", "rho", "aggregate_primary_violation")}
            for slot in range(config.slots):
                inputs = [observation.reshape(-1)] if shared else observation
                values = [policy(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))[0]
                          for policy, state in zip(policies, inputs)]
                chosen = [rngs[i].randrange(len(actions)) if random_policy or i == random_agent else int(q.argmax())
                          for i, q in enumerate(values)]
                if shared:
                    chosen = chosen * config.num_su
                    values = values * config.num_su
                for i, q in enumerate(values):
                    selected_q[i].append(float(q[chosen[i]]))
                observation, reward, _, info = env.step(chosen)
                batteries.append(info["battery"])
                if retain_slots:
                    slots_retained.append({"episode": episode, **info})
                mapping = dict(reward="rewards", score="scores", executed="executed", attempted="attempted_transmit",
                    battery_failed="battery_constraint_failed", primary_failed="primary_constraint_failed",
                    debit="debit", battery="battery", harvest="harvest")
                for key, source in mapping.items():
                    totals[key] += info[source]
                tuples = np.asarray(info["actions"])
                totals["power"] += tuples[:, 1]
                totals["rho"] += tuples[:, 2]
                v = info["primary"]
                aggregate = np.sum(np.asarray(info["executed"])*(1-tuples[:,2])*tuples[:,1]*env.trace["sp"][slot,:,v])
                totals["aggregate_primary_violation"] += int(aggregate > config.primary_threshold)
            rows.append({"episode": episode, **{k: v.tolist() for k, v in totals.items()}})
    after = training_fingerprint(agents)
    if before != after:
        raise AssertionError("frozen evaluation mutated populated training state")
    bounds = q_bounds(config, actions, bank, agents[0].config.gamma)
    if shared:
        bounds = [bound * config.num_su for bound in bounds]
    qstats = [{"min": float(np.min(q)), "p99": float(np.quantile(q, .99)), "max": float(np.max(q))} for q in selected_q]
    finite = all(np.isfinite(list(s.values())).all() for s in qstats) and all(
        np.isfinite(v).all() for row in rows for k, v in row.items() if k != "episode")
    healthy = finite and all(s["min"] >= 1.1*bounds[0] and s["p99"] <= 1.1*bounds[1] for s in qstats)
    means = {key: np.mean([r[key] for r in rows], axis=0).tolist() for key in rows[0] if key != "episode"}
    reward = np.asarray(means["reward"])/config.slots
    fairness = float(reward.sum()**2/(len(reward)*np.square(reward).sum())) if np.square(reward).sum() else 1.0
    return dict(rows=rows, mean_per_episode=means, selected_q=qstats, q_bounds=bounds,
                selected_q_values=selected_q, slot_records=slots_retained,
                battery_quantiles={str(q): np.quantile(batteries,q,axis=0).tolist() for q in (.05,.5,.95)},
                q_reward_scope="network_sum" if shared else "own_su",
                healthy=healthy, finite=bool(finite), training_before=before, training_after=after,
                fairness_jain_signed_reward=fairness)


def admissibility(finals, randoms):
    checks = dict(q_health=all(x["healthy"] for x in finals),
        score=bool(np.mean([x["mean_per_episode"]["score"] for x in finals]) >
                   np.mean([x["mean_per_episode"]["score"] for x in randoms])),
        every_su_execution=bool(np.all(np.sum([x["mean_per_episode"]["executed"] for x in finals],axis=0)>0)))
    return {"passed": all(checks.values()), "checks": checks}


def screen_promotion(final, baseline, incumbent_reward, slots):
    eligibility = admissibility([final], [baseline])
    gain = float(np.mean(final["mean_per_episode"]["reward"])/slots-incumbent_reward)
    return {"accepted": bool(eligibility["passed"] and gain >= .05), "gain": gain,
            "eligibility": eligibility}


def ensure_training_allowed(ledger):
    if ledger.get("locked"):
        raise RuntimeError("training prohibited after audit lock")


def ensure_audit_ready(ledger, completion):
    if not completion or not completion.get("complete"):
        raise RuntimeError("durable study completion required before audit")
    if any(cell.get("status") != "complete" for cell in ledger["cells"]):
        raise RuntimeError("unfinished training cells prohibit audit")
    if len(ledger["cells"]) > 31 or ledger["training_seconds"] > 3600:
        raise RuntimeError("study budget invalid")


def wins(candidate, incumbent, margin=.05):
    differences = np.asarray(candidate)-np.asarray(incumbent)
    return bool(differences.mean() >= margin and np.sum(differences > 0) >= 2)


def select_candidate(anchor, plain, double):
    """Inputs carry healthy and three monitor rewards per SU-slot; ties retain."""
    chosen, reason = ("anchor", "healthy anchor retained") if anchor["healthy"] else (None, "no healthy anchor")
    if plain["healthy"] and (chosen is None or wins(plain["reward"], anchor["reward"])):
        chosen = "plain"
        reason = "plain health fallback" if not anchor["healthy"] else "plain gain and wins"
    if double["healthy"] and wins(double["reward"], plain["reward"]):
        if chosen is None or chosen == "plain" or wins(double["reward"], anchor["reward"]):
            chosen, reason = "double", "Double DQN gain and wins"
    return {"selected": chosen, "reason": reason}


def learning_gate(finals, initials, randoms, slots):
    f = np.array([x["mean_per_episode"]["reward"] for x in finals])/slots
    initial = np.array([x["mean_per_episode"]["reward"] for x in initials])/slots
    baseline = np.array([x["mean_per_episode"]["reward"] for x in randoms])/slots
    score = np.mean([x["mean_per_episode"]["score"] for x in finals])
    random_score = np.mean([x["mean_per_episode"]["score"] for x in randoms])
    executions = np.sum([x["mean_per_episode"]["executed"] for x in finals], axis=0)
    checks = dict(q_health=all(x["healthy"] for x in finals), reward_random=float((f-baseline).mean()) >= .05,
        reward_initial=float((f-initial).mean()) >= .05,
        wins_random=int(np.sum((f-baseline).mean(axis=1)>0)) >= 2,
        wins_initial=int(np.sum((f-initial).mean(axis=1)>0)) >= 2,
        every_su_reward=bool(np.all((f-baseline).mean(axis=0)>0)), score=bool(score>random_score),
        every_su_execution=bool(np.all(executions>0)))
    return {"passed": all(checks.values()), "checks": checks,
            "reward_advantage_per_su_slot": (f-baseline).mean(axis=0).tolist(),
            "score_advantage_per_su_slot": float((score-random_score)/slots)}
