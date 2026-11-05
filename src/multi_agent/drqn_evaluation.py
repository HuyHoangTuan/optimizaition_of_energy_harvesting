"""Frozen evaluation with local recurrent memory and populated-state checks."""
import random

import numpy as np
import torch

from .environment import JointEnvironment
from .recurrent import RecurrentQNetwork
from .stability import fingerprint, training_fingerprint, q_bounds, learning_gate


def populated_fingerprint(agents):
    return fingerprint(dict(base=training_fingerprint(agents), recurrent=[
        dict(hidden=getattr(a, 'hidden', None), prefix=getattr(a, 'acting_prefix', None),
             pending=getattr(a, 'pending', None),
             mask_rng=a.mask_rng.getstate() if hasattr(a, 'mask_rng') else None)
        for a in agents]))


def evaluate(agents, config, actions, bank, seed, *, random_policy=False,
             state_evidence='populated_training_state'):
    before = populated_fingerprint(agents)
    env = JointEnvironment(config, actions=actions, trace_bank=bank)
    rngs = [random.Random(seed + i*1009) for i in range(config.num_su)]
    rows, slots, batteries = [], [], []
    selected_q = [[] for _ in agents]
    with torch.no_grad():
        for episode in range(len(bank)):
            observation = env.reset()
            hidden = [None for _ in agents]
            totals = {key: np.zeros(config.num_su) for key in ('reward', 'score', 'executed',
                      'attempted', 'battery_failed', 'primary_failed', 'debit', 'battery', 'harvest')}
            for _ in range(config.slots):
                values = []
                for i, a in enumerate(agents):
                    state = torch.as_tensor(observation[i], dtype=torch.float32)
                    if isinstance(a.policy, RecurrentQNetwork):
                        q, hidden[i] = a.policy(state[None, None], hidden[i])
                        values.append(q[0, 0])
                    else:
                        values.append(a.policy(state[None])[0])
                chosen = [rngs[i].randrange(len(actions)) if random_policy else int(q.argmax())
                          for i, q in enumerate(values)]
                for i, q in enumerate(values):
                    selected_q[i].append(float(q[chosen[i]]))
                observation, _, _, info = env.step(chosen)
                slots.append(dict(episode=episode, **info))
                batteries.append(info['battery'])
                for key, source in dict(reward='rewards', score='scores', executed='executed',
                    attempted='attempted_transmit', battery_failed='battery_constraint_failed',
                    primary_failed='primary_constraint_failed', debit='debit', battery='battery', harvest='harvest').items():
                    totals[key] += info[source]
            rows.append(dict(episode=episode, **{key: value.tolist() for key,value in totals.items()}))
    after = populated_fingerprint(agents)
    if before != after:
        raise AssertionError('evaluation mutated training state')
    bounds = q_bounds(config, actions, bank, agents[0].config.gamma)
    stats = [dict(min=float(np.min(q)), p99=float(np.quantile(q,.99)), max=float(np.max(q))) for q in selected_q]
    finite = bool(all(np.isfinite(v).all() for row in rows for key,v in row.items() if key != 'episode')
                  and all(np.isfinite(list(stat.values())).all() for stat in stats))
    healthy = finite and all(s['min'] >= 1.1*bounds[0] and s['p99'] <= 1.1*bounds[1] for s in stats)
    return dict(rows=rows, slot_records=slots,
                mean_per_episode={key: np.mean([row[key] for row in rows], axis=0).tolist()
                                  for key in rows[0] if key != 'episode'},
                selected_q=stats, selected_q_values=selected_q, q_bounds=bounds,
                battery_quantiles={str(q): np.quantile(batteries,q,axis=0).tolist() for q in (.05,.5,.95)},
                finite=finite, healthy=bool(healthy), training_before=before, training_after=after,
                state_evidence=state_evidence, q_reward_scope='own_su')


def stage_gate(finals, initials, randoms, slots=20):
    result = learning_gate(finals, initials, randoms, slots)
    result['checks']['finite'] = all(row['finite'] for row in finals)
    result['passed'] = all(result['checks'].values())
    return result
