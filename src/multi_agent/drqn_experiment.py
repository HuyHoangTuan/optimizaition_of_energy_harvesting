"""Guarded experiment stages. No work runs on import and no legacy runner loads."""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time
import uuid

import numpy as np
import torch

from .agent import AgentConfig, IndependentDQN
from .environment import EnvironmentConfig, JointEnvironment
from .recurrent import IndependentDRQN
from .drqn_evaluation import evaluate, stage_gate
from .drqn_traces import (source_bank, load_bank, save_bank, bank_identity, reject_overlap,
                          checked_legacy_constructor)

PROTOCOL_SHA256 = '95bb3fa433489fcf7e4bb3792264bb3c9495e8f95eb744d7fa7fc698c6686027'
PROJECT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def append(path, value):
    with Path(path).open('a', encoding='utf8') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                packaging=importlib.metadata.version('packaging'),
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                device='cpu', platform=platform.platform())


def configure_runtime():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def scientific_sources():
    # Freeze all local modules, including legacy source used to check provenance.
    return [p for p in sorted((PROJECT/'src').rglob('*.py')) if p.name != 'drqn_analysis.py'] + [
        PROJECT/'run_drqn.py',PROJECT/'requirements-multi-agent.txt']


def manifest_value(root):
    inputs = sorted((root/'inputs').rglob('*'))
    return dict(schema=1, sources={str(p.relative_to(PROJECT)).replace('\\','/'):sha(p)
                                 for p in scientific_sources()},
                inputs={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in inputs if p.is_file()},
                runtime=runtime())


def verify_identity(root):
    root = Path(root)
    try:
        expected = read(root/'freeze.json')
        current = manifest_value(root)
        if expected != current:
            raise RuntimeError('frozen scientific source/input/runtime identity changed')
        return sha(root/'freeze.json')
    except Exception as exc:
        write_new(root/'rejections'/(uuid.uuid4().hex+'.json'),
                  dict(kind='identity_rejection', reason=str(exc), runtime=runtime()))
        raise


def freeze(root):
    root = Path(root)
    if (root/'freeze.json').exists():
        return verify_identity(root)
    if not (root/'preflight.json').exists():
        raise RuntimeError('successful preflight required before freeze')
    write_new(root/'freeze.json', manifest_value(root))
    return sha(root/'freeze.json')


def preflight(root, protocol_path, legacy_root, legacy_manifest_path, *, do_freeze=False):
    root, legacy_root = Path(root), Path(legacy_root)
    if (root/'preflight.json').exists():
        if not do_freeze:
            raise FileExistsError('preflight already exists; outputs are never overwritten')
        if sha(protocol_path) != sha(root/'inputs/protocol.json'):
            raise RuntimeError('protocol differs from completed preflight')
        return freeze(root)
    if sha(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError('reviewed protocol identity mismatch')
    root.mkdir(parents=True, exist_ok=False)
    inputs = root/'inputs'; inputs.mkdir()
    shutil.copyfile(protocol_path, inputs/'protocol.json')
    shutil.copyfile(legacy_manifest_path, inputs/'legacy-artifact-sha256.json')
    provenance = read(legacy_manifest_path)
    if sha(legacy_root/'protocol.json') != provenance['protocol.json']['sha256']:
        raise RuntimeError('legacy protocol provenance mismatch')
    legacy_protocol = read(legacy_root/'protocol.json')
    for rel in ('src/modules/environment/Environment.py','src/utils/RandomUtils.py'):
        expected = legacy_protocol['legacy_source_sha256'].get(rel.replace('/','\\'))
        if sha(PROJECT/rel) != expected:
            raise RuntimeError('legacy constructor source provenance mismatch: '+rel)
    shutil.copyfile(legacy_root/'protocol.json',inputs/'legacy-protocol.json')
    protocol = read(protocol_path)
    identities = {}
    started = time.perf_counter()
    for n in protocol['su_stage_order']:
        banks = {}
        for seed in protocol['seeds']:
            rel = f'traces/n{n}_fixed_seed{seed}_e1600.npz'
            source = legacy_root/rel
            if rel not in provenance or sha(source) != provenance[rel]['sha256']:
                raise RuntimeError('legacy bank provenance mismatch: '+rel)
            bank = load_bank(source)
            generated, actions = source_bank(n, seed, 1600)
            if bank_identity(bank) != bank_identity(generated):
                raise RuntimeError('translated source constructor differs from frozen bank')
            reference_bank, reference_actions = checked_legacy_constructor(PROJECT,n,seed,1600)
            if bank_identity(reference_bank) != bank_identity(bank) or np.asarray(reference_actions).tolist()!=np.asarray(actions).tolist():
                raise RuntimeError('exact checked legacy constructor bank/map differs')
            # Check action-map provenance against prior per-cell metadata.
            name = f'anchor_seed{seed}' if n == 1 else f'independent_n{n}_seed{seed}'
            meta_rel = f'runs/{name}/metadata.json'
            metadata = legacy_root/meta_rel
            if meta_rel in provenance and sha(metadata) != provenance[meta_rel]['sha256']:
                raise RuntimeError('legacy map provenance mismatch: '+meta_rel)
            if np.asarray(read(metadata)['action_map']).tolist() != np.asarray(actions).tolist():
                raise RuntimeError('source map differs from prior frozen map')
            target = inputs/f'train_n{n}_seed{seed}.npz'
            shutil.copyfile(source, target)
            write_new(inputs/f'actions_n{n}_seed{seed}.json', actions)
            shutil.copyfile(metadata,inputs/f'legacy_metadata_n{n}_seed{seed}.json')
            write_new(inputs/f'provenance_n{n}_seed{seed}.json',
                      dict(bank_key=rel, bank_sha256=sha(source), map_key=meta_rel,
                           metadata_sha256=sha(metadata), constructor_exact_canonical_match=True))
            banks[f'train_{seed}'] = bank
        for label, seed, episodes in [('monitor',protocol['monitor_seed'],100),
                                      ('pilot',protocol['pilot_seed'],100)]:
            bank, actions = source_bank(n,seed,episodes)
            save_bank(inputs/f'{label}_n{n}.npz',bank)
            if label == 'pilot':
                write_new(inputs/f'actions_pilot_n{n}.json', actions)
            banks[label] = bank
        identities[str(n)] = reject_overlap(banks)
    report = dict(status='preflight_only_no_training', runtime=runtime(),
                  protocol_sha256=sha(protocol_path), bank_identities=identities,
                  generation_and_validation_seconds=time.perf_counter()-started,
                  parameter_counts={name:sum(p.numel() for p in make_agents(name,1,2)[0].policy.parameters())
                                    for name in ('dqn','drqn')},
                  freeze_status='not_frozen_until_explicit_post_verification_freeze')
    write_new(root/'preflight.json', report)
    if do_freeze:
        freeze(root)
    return report


class BudgetStop(RuntimeError):
    pass


class Ledger:
    """Append-only events charge every measured step, including exceptions."""
    def __init__(self, root, cap=3600., cell_cap=18):
        self.root, self.cap, self.cell_cap = Path(root), cap, cell_cap
        self.path = self.root/'ledger.jsonl'
        self._total = sum(e.get('seconds',0.) for e in self.events() if e['event']=='charge')

    def events(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def total(self):
        return self._total

    def allowed(self):
        if (self.root/'seal.json').exists() or (self.root/'audit-open.json').exists():
            raise RuntimeError('training prohibited after durable training seal/audit open')
        if self.total() >= self.cap:
            raise BudgetStop('cumulative training cap reached')

    def start(self, name, kind, identity):
        self.allowed()
        events = self.events()
        starts = [e for e in events if e['event']=='start']
        ended = {e['name'] for e in events if e['event']=='finish'}
        if any(e['name'] not in ended for e in starts):
            raise RuntimeError('unfinished attempt blocks training; preserve and recover its timing first')
        if any(e['name']==name for e in starts):
            raise FileExistsError('duplicate attempt name')
        if kind=='cell' and sum(e['kind']=='cell' for e in starts) >= self.cell_cap:
            raise BudgetStop('full-cell attempt cap reached')
        append(self.path,dict(event='start', name=name, kind=kind, identity=identity,
                              utc=datetime.now(timezone.utc).isoformat()))

    @contextmanager
    def step(self, name):
        self.allowed()
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter()-started
            append(self.path,dict(event='charge',name=name,seconds=elapsed))
            self._total += elapsed

    def finish(self, name, status, **fields):
        append(self.path,dict(event='finish',name=name,status=status,**fields))

    def overhead(self, name, category, seconds):
        append(self.path,dict(event='overhead',name=name,category=category,seconds=seconds))


def forecast(remaining, per_step, spent, cap=3600.):
    seconds = 2 * sum(steps * per_step[learner] for learner,steps in remaining)
    return dict(forecast_seconds=seconds, remaining_seconds=cap-spent,
                admitted=bool(seconds <= cap-spent), safety_factor=2)


def make_agents(learner,n,seed):
    if learner not in ('dqn','drqn') or n not in (1,2,3):
        raise ValueError('frozen learner and SU stage required')
    cls = IndependentDQN if learner=='dqn' else IndependentDRQN
    config = AgentConfig.legacy_parity(**({} if learner=='dqn' else {'hidden':64}))
    return [cls(i,9+n,32,seed+1009*i,config) for i in range(n)]


def environment(n, seed, bank, actions):
    return JointEnvironment(EnvironmentConfig(num_su=n, seed=seed, debit_mode='requested'),
                            trace_bank=bank, actions=actions)


def save_checkpoints(folder, agents, episode, identity):
    hashes = {}
    for a in agents:
        path = folder/f'su{a.su_id}_ep{episode}.pt'
        with path.open('xb') as stream:
            torch.save({**a.checkpoint(), 'freeze_sha256':identity},stream)
        hashes[path.name] = sha(path)
    return hashes


def workload(root, learner, n, seed, *, kind='cell', episodes=1600):
    """One attempt, never resumed or overwritten; all partial evidence survives."""
    attempt_started=time.perf_counter()
    root = Path(root)
    identity = verify_identity(root)
    ledger = Ledger(root)
    charged_before=ledger.total()
    name = f'{kind}_{learner}_n{n}_seed{seed}'
    ledger.start(name,kind,identity)
    folder = root/'attempts'/name
    folder.mkdir(parents=True, exist_ok=False)
    agents = make_agents(learner,n,seed)
    inputs = root/'inputs'
    bank = load_bank(inputs/(f'pilot_n{n}.npz' if kind=='pilot' else f'train_n{n}_seed{seed}.npz'))
    actions = read(inputs/(f'actions_pilot_n{n}.json' if kind=='pilot' else f'actions_n{n}_seed{seed}.json'))
    env = environment(n,seed,bank,actions)
    monitor = load_bank(inputs/f'monitor_n{n}.npz')
    metadata = dict(learner=learner,n=n,seed=seed,kind=kind,episodes=episodes,freeze_sha256=identity,
                    action_map=actions,environment=asdict(env.config),
                    parameters=[sum(p.numel() for p in a.policy.parameters()) for a in agents],
                    checkpoint_contract='evaluation only; no exact resume')
    write_new(folder/'metadata.json', metadata)
    checkpoints, evaluations = {}, {}
    step_costs, postwarm, completed_steps = [], [], 0
    failure = None; status = 'failed'
    def checkpoint(ep):
        nonlocal checkpoints
        verify_identity(root)
        started = time.perf_counter()
        checkpoints[str(ep)] = save_checkpoints(folder,agents,ep,identity)
        evaluations[str(ep)] = evaluate(agents,env.config,actions,monitor,920003,
            state_evidence='initial_empty_training_state' if ep==0 else 'populated_training_state')
        write_new(folder/f'monitor_ep{ep}.json',evaluations[str(ep)])
        ledger.overhead(name,'checkpoint_and_monitor',time.perf_counter()-started)
    try:
        if kind=='cell':
            checkpoint(0)
            started = time.perf_counter()
            verify_identity(root)
            random_result = evaluate(agents,env.config,actions,monitor,920003,random_policy=True)
            write_new(folder/'monitor_random.json',random_result)
            ledger.overhead(name,'random_monitor',time.perf_counter()-started)
        for ep in range(episodes):
            # Reset is measured with the first collection step.
            observations = None
            for slot in range(20):
                elapsed_before = ledger.total()
                with ledger.step(name):
                    if slot==0:
                        observations = env.reset()
                        for a in agents:
                            if hasattr(a,'reset_hidden'): a.reset_hidden()
                    chosen = [a.select_action(observations[i]) for i,a in enumerate(agents)]
                    following,rewards,terminal,info = env.step(chosen)
                    losses = []
                    for i,a in enumerate(agents):
                        a.remember(observations[i],chosen[i],rewards[i],following[i],terminal)
                        losses.append(a.optimize())
                    if terminal:
                        for a in agents: a.end_episode()
                    collected_observations=observations
                    observations = following
                cost = ledger.total()-elapsed_before
                step_costs.append(cost)
                if all(a.updates>0 for a in agents): postwarm.append(cost)
                completed_steps += 1
                record = dict(episode=ep,**info,terminal=terminal,losses=losses,
                              observations=collected_observations.tolist(),next_observations=following.tolist(),
                              diagnostics=[a.last_diagnostics for a in agents])
                started=time.perf_counter(); append(folder/'slots.jsonl',record)
                ledger.overhead(name,'raw_serialization',time.perf_counter()-started)
                if ledger.total() >= ledger.cap:
                    raise BudgetStop('actual cap reached after joint step')
            if kind=='cell' and ep+1 in (400,800,1200,1600): checkpoint(ep+1)
        status='complete'
    except BudgetStop as exc:
        status='partial'; failure=str(exc)
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
    finally:
        if status!='complete':
            started=time.perf_counter()
            checkpoints['partial'] = save_checkpoints(folder,agents,'partial',identity)
            ledger.overhead(name,'partial_checkpoint',time.perf_counter()-started)
        result = dict(status=status,failure=failure,completed_steps=completed_steps,
                      updates=[a.updates for a in agents],checkpoints=checkpoints,
                      step_seconds=step_costs,postwarm_seconds_per_step=float(np.mean(postwarm)) if postwarm else None,
                      freeze_sha256=identity)
        result['attempt_wall_seconds']=time.perf_counter()-attempt_started
        result['training_seconds']=ledger.total()-charged_before
        result['overhead_seconds']=result['attempt_wall_seconds']-result['training_seconds']
        write_new(folder/'result.json',result)
        ledger.finish(name,status,completed_steps=completed_steps,failure=failure)
    return result


def benchmark(root,learner,n):
    attempt_started=time.perf_counter()
    root=Path(root); identity=verify_identity(root); ledger=Ledger(root)
    name=f'benchmark_{learner}_n{n}'
    ledger.start(name,'benchmark',identity)
    charged_before=ledger.total()
    folder=root/'attempts'/name; folder.mkdir(parents=True,exist_ok=False)
    agents=make_agents(learner,n,1000007)
    bank=load_bank(root/'inputs'/f'train_n{n}_seed2.npz')
    actions=read(root/'inputs'/f'actions_n{n}_seed2.json')
    env=environment(n,2,bank,actions)
    costs=[]; status='failed'; error=None
    try:
        # Populate real late replay. This collection and its random-step hidden work
        # are charged; no optimization is hidden outside the training timer.
        for ep in range(1600):
            for slot in range(20):
                with ledger.step(name):
                    if slot==0:observations=env.reset()
                    chosen=[a.select_action(observations[i]) for i,a in enumerate(agents)]
                    following,rewards,terminal,_=env.step(chosen)
                    for i,a in enumerate(agents):
                        a.remember(observations[i],chosen[i],rewards[i],following[i],terminal)
                    observations=following
                    if terminal:
                        for a in agents: a.end_episode()
        # Each timed joint transition starts from an actual late-episode prefix.
        # Fixture construction is charged separately and never included in the
        # representative per-step denominator.
        for iteration in range(64):
            for slot in range(18):
                with ledger.step(name):
                    if slot==0:
                        observations=env.reset(iteration)
                        for a in agents:
                            if hasattr(a,'reset_hidden'):a.reset_hidden()
                    chosen=[]
                    for i,a in enumerate(agents):
                        if learner=='drqn':
                            with torch.no_grad():
                                state=a._observation(observations[i]);q,h=a.policy(torch.from_numpy(state)[None,None],a.hidden)
                            a.hidden=h.detach();a.acting_prefix.append(state);chosen.append(int(q.argmax()))
                        else:
                            with torch.no_grad():chosen.append(int(a.policy(torch.as_tensor(observations[i],dtype=torch.float32)).argmax()))
                    following,rewards,terminal,_=env.step(chosen)
                    if learner=='drqn':
                        for i,a in enumerate(agents):
                            a.remember(observations[i],chosen[i],rewards[i],following[i],terminal)
                    observations=following
            before=ledger.total()
            with ledger.step(name):
                chosen=[]
                for i,a in enumerate(agents):
                    if learner=='drqn':
                        state=a._observation(observations[i])
                        with torch.no_grad():q,h=a.policy(torch.from_numpy(state)[None,None],a.hidden)
                        a.hidden=h.detach();a.acting_prefix.append(state);chosen.append(int(q.argmax()))
                    else:
                        with torch.no_grad():chosen.append(int(a.policy(torch.as_tensor(observations[i],dtype=torch.float32)).argmax()))
                following,rewards,terminal,_=env.step(chosen)
                for i,a in enumerate(agents):
                    if learner=='dqn':a.remember(observations[i],chosen[i],rewards[i],following[i],terminal)
                    if learner=='drqn':
                        a.remember(observations[i],chosen[i],rewards[i],following[i],False)
                    a.optimize()
            costs.append(ledger.total()-before)
            for a in agents:
                if learner=='drqn':a.pending.clear();a.reset_hidden()
        status='complete'
    except BudgetStop as exc:status='partial';error=str(exc)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    result=dict(status=status,error=error,representative_steps=len(costs),joint_step_seconds=costs,
                seconds_per_joint_step=float(np.mean(costs)) if costs else None,
                replay_occupancy=[len(a.replay) for a in agents],prefix_before_measured_step=18,
                freeze_sha256=identity)
    result['attempt_wall_seconds']=time.perf_counter()-attempt_started
    result['training_seconds']=ledger.total()-charged_before
    result['overhead_seconds']=result['attempt_wall_seconds']-result['training_seconds']
    write_new(folder/'result.json',result);ledger.finish(name,status,error=error)
    return result


def pilot(root):
    root=Path(root);verify_identity(root)
    for learner in ('dqn','drqn'):
        result=workload(root,learner,1,1000007,kind='pilot',episodes=100)
        if result['status']!='complete':return result
    return dict(status='complete')


def study(root):
    root=Path(root);verify_identity(root);ledger=Ledger(root)
    ledger.allowed()
    for n in (1,2,3):
        if n>1:
            prior=read(root/f'stage_n{n-1}.json')
            if not prior['passed']:break
        rates={}
        for learner in ('dqn','drqn'):
            pilot_result=read(root/'attempts'/f'pilot_{learner}_n1_seed1000007/result.json')
            if pilot_result['status']!='complete':raise RuntimeError('both complete pilots required')
            bench_path=root/'attempts'/f'benchmark_{learner}_n{n}/result.json'
            bench=read(bench_path) if bench_path.exists() else benchmark(root,learner,n)
            if bench['status']!='complete':return dict(status='benchmark_incomplete')
            rates[learner]=max(pilot_result['postwarm_seconds_per_step']*n,bench['seconds_per_joint_step'])
        cells=[(learner,seed) for learner in ('dqn','drqn') for seed in (2,17,42)]
        # Recheck all remaining stage cells before every cell boundary.
        for index,(learner,seed) in enumerate(cells):
            # benchmark/workload use separate ledger instances; reload actual charges.
            ledger=Ledger(root)
            admission=forecast([(name,32000) for name,_ in cells[index:]],rates,ledger.total())
            append(root/'admissions.jsonl',dict(n=n,learner=learner,seed=seed,rates=rates,**admission))
            if not admission['admitted']:
                write_new(root/f'stage_n{n}.json',dict(passed=False,status='budget_not_admitted',admission=admission))
                seal(root,'budget_not_admitted');return admission
            result=workload(root,learner,n,seed)
            if result['status']!='complete':
                write_new(root/f'stage_n{n}.json',dict(passed=False,status=result['status']))
                seal(root,'unexpected_partial_or_failure');return result
        gates={}
        for learner in ('dqn','drqn'):
            folders=[root/'attempts'/f'cell_{learner}_n{n}_seed{seed}' for seed in (2,17,42)]
            gates[learner]=stage_gate([read(p/'monitor_ep1600.json') for p in folders],
                                      [read(p/'monitor_ep0.json') for p in folders],
                                      [read(p/'monitor_random.json') for p in folders])
        decision=dict(passed=all(g['passed'] for g in gates.values()),gates=gates,status='complete')
        write_new(root/f'stage_n{n}.json',decision)
        if not decision['passed']:break
    seal(root,'gated_study_complete')
    return dict(status='sealed')


def seal(root, reason):
    root=Path(root);identity=verify_identity(root)
    if not (root/'stage_n1.json').exists():
        raise RuntimeError('durable first-stage decision required before training seal')
    events=Ledger(root).events()
    starts={e['name'] for e in events if e['event']=='start'}
    finished={e['name'] for e in events if e['event']=='finish'}
    if starts!=finished:raise RuntimeError('unfinished attempt prevents durable seal')
    required=list((root/'attempts').rglob('*'))+list(root.glob('stage_n*.json'))
    files={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in required if p.is_file()}
    files['ledger.jsonl']=sha(root/'ledger.jsonl')
    if (root/'admissions.jsonl').exists():files['admissions.jsonl']=sha(root/'admissions.jsonl')
    write_new(root/'seal.json',dict(reason=reason,freeze_sha256=identity,files=files,
                                  training_seconds=Ledger(root).total()))


def verify_seal(root):
    root=Path(root);identity=verify_identity(root);value=read(root/'seal.json')
    if 'stage_n1.json' not in value['files']:
        raise RuntimeError('sealed stage decisions missing')
    if identity!=value['freeze_sha256']:raise RuntimeError('seal identity mismatch')
    if any(sha(root/rel)!=expected for rel,expected in value['files'].items()):
        raise RuntimeError('sealed training artifact changed')
    if Ledger(root).total()>3600:
        # A last joint step may cross the cap. Preserve that partial, but reject
        # the study rather than representing it as compliant budget evidence.
        raise RuntimeError('training cap exceeded; audit acceptance blocked')
    return value


def audit(root):
    root=Path(root);verify_seal(root)
    protocol=read(root/'inputs/protocol.json')
    # Exclusive durable record exists before audit-bank construction or access.
    write_new(root/'audit-open.json',dict(seal_sha256=sha(root/'seal.json'),
                                        rule='training permanently prohibited'))
    folder=root/'audit';folder.mkdir(exist_ok=False)
    results={}
    for n in (1,2,3):
        cells=[root/'attempts'/f'cell_{learner}_n{n}_seed{seed}'
               for learner in ('dqn','drqn') for seed in (2,17,42)]
        complete=[p for p in cells if (p/'result.json').exists() and read(p/'result.json')['status']=='complete']
        if not complete:continue
        bank,_=source_bank(n,protocol['audit_seed'],protocol['evaluation_episodes'])
        training={f'train_{seed}':load_bank(root/'inputs'/f'train_n{n}_seed{seed}.npz') for seed in (2,17,42)}
        identities=reject_overlap(dict(training,monitor=load_bank(root/'inputs'/f'monitor_n{n}.npz'),audit=bank))
        save_bank(folder/f'bank_n{n}.npz',bank)
        write_new(folder/f'bank_n{n}_identities.json',identities)
        for cell in complete:
            verify_identity(root)
            meta=read(cell/'metadata.json');agents=make_agents(meta['learner'],n,meta['seed'])
            actions=meta['action_map'];config=EnvironmentConfig(**meta['environment'])
            evaluations={}
            for ep in (0,1600):
                for a in agents:
                    state=torch.load(cell/f'su{a.su_id}_ep{ep}.pt',map_location='cpu')
                    a.policy.load_state_dict(state['policy'])
                verify_identity(root)
                evaluations[str(ep)]=evaluate(agents,config,actions,bank,920003,
                            state_evidence='reconstructed_empty_replay_evaluation_only')
            verify_identity(root)
            evaluations['random']=evaluate(agents,config,actions,bank,920003,random_policy=True,
                            state_evidence='reconstructed_empty_replay_evaluation_only')
            write_new(folder/(cell.name+'.json'),evaluations)
            results[cell.name]=dict(reward_per_su_slot=float(np.mean(evaluations['1600']['mean_per_episode']['reward'])/20))
    write_new(folder/'summary.json',results)
    return results


def verify(root):
    root=Path(root);identity=verify_identity(root)
    if (root/'seal.json').exists():verify_seal(root)
    events=Ledger(root).events()
    starts=[e for e in events if e['event']=='start'];finishes=[e for e in events if e['event']=='finish']
    if len({e['name'] for e in starts})!=len(starts):raise AssertionError('duplicate attempts')
    if {e['name'] for e in starts}!={e['name'] for e in finishes}:raise AssertionError('unfinished attempts')
    if Ledger(root).total()>3600:raise AssertionError('training cap exceeded')
    if sum(e['kind']=='cell' for e in starts)>18:raise AssertionError('cell cap exceeded')
    return dict(status='local_identity_and_ledger_checks_passed',freeze_sha256=identity,
                training_seconds=Ledger(root).total(),attempts=len(starts),
                independent_verification='required separately; this command is not independent review')
