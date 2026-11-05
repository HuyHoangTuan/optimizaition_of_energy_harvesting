"""Portable translation of the legacy constructor's trace draw order.

The unused g_p draws are retained. Channel arrays are density grids, not iid
channel samples. No legacy module initializer or logging code is imported.
"""
import hashlib
import builtins
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FIELDS = ('rt', 'sp', 'pr', 'ps', 'primary_power', 'ambient')


def checked_legacy_constructor(project, num_su, seed, episodes):
    """Execute verified source in private namespaces, without package initializers.

    The caller checks original file hashes before this call. The RandomUtils
    module receives a private Python RNG. Logging receives a no-op object.
    Imports and global caller RNG states remain unchanged.
    """
    project=Path(project)
    python_rng=random.Random(seed)
    ru_globals={'__builtins__':dict(vars(builtins))}
    def import_ru(name,*args,**kwargs):
        return python_rng if name=='random' else builtins.__import__(name,*args,**kwargs)
    ru_globals['__builtins__']['__import__']=import_ru
    ru_path=project/'src/utils/RandomUtils.py'
    exec(compile(ru_path.read_text(encoding='utf8'),str(ru_path),'exec'),ru_globals)
    # The original file seeds to 2 at import; apply the requested experiment seed.
    python_rng.seed(seed);ru_globals['RandomUtils']=np.random.default_rng(seed)
    ru=SimpleNamespace(**{name:value for name,value in ru_globals.items() if not name.startswith('__')})
    utils=SimpleNamespace(RandomUtils=ru,LogUtils=SimpleNamespace(info=lambda *args,**kwargs:None))
    env_globals={'__builtins__':dict(vars(builtins))}
    def import_env(name,*args,**kwargs):
        return utils if name=='src.utils' else builtins.__import__(name,*args,**kwargs)
    env_globals['__builtins__']['__import__']=import_env
    env_path=project/'src/modules/environment/Environment.py'
    exec(compile(env_path.read_text(encoding='utf8'),str(env_path),'exec'),env_globals)
    env=env_globals['Environment'](NumSU=num_su,Episode=episodes,Dynamic_Rho=False)
    bank=[dict(**{key:np.asarray(getattr(env,old)[e]).transpose(2,0,1).copy()
            for key,old in [('rt','g_rt'),('sp','g_sp'),('pr','g_pr'),('ps','g_ps'),('primary_power','_P_p')]},
               ambient=np.asarray(env._E_ambient[e]).copy()) for e in range(episodes)]
    return bank,env.actions_space


def source_bank(num_su, seed, episodes):
    rng = np.random.default_rng(seed)
    def grid():
        exp = rng.exponential(1., 20)
        rate = min(1 / np.mean(exp), 1.)
        return rate * np.exp(-rate * np.linspace(exp.min(), exp.max(), 20))
    diagonal = np.array([[grid() for _ in range(num_su)] for _ in range(episodes)])
    channels = {}
    for field in ('pr', 'sp', 'ps'):
        channels[field] = np.array([[[grid() for _ in range(2)] for _ in range(num_su)]
                                    for _ in range(episodes)]).transpose(0, 3, 1, 2)
    # The original constructor consumes unused primary-primary channel draws.
    for _ in range(episodes):
        for _ in range(2):
            grid()
    power = np.array([[[rng.uniform(.001, 1., 20) for _ in range(2)] for _ in range(num_su)]
                      for _ in range(episodes)]).transpose(0, 3, 1, 2)
    ambient = np.array([rng.uniform(.001, .2, 20) for _ in range(episodes)])
    rt = np.array([[[diagonal[e, i] if i == j else grid() for j in range(num_su)]
                     for i in range(num_su)] for e in range(episodes)]).transpose(0, 3, 1, 2)
    arrays = dict(rt=rt, primary_power=power, ambient=ambient, **channels)
    # The legacy constructor shuffles the meshgrid map with Python's seeded RNG.
    actions = [tuple(v) for v in np.array(np.meshgrid([0, 1], [i/16 for i in range(1,17)], [.4])).T.reshape(-1,3)]
    random.Random(seed).shuffle(actions)
    return [{key: arrays[key][e].copy() for key in FIELDS} for e in range(episodes)], actions


def save_bank(path, bank):
    # Exclusive creation also protects files when callers reserve only a parent.
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **{key: np.stack([row[key] for row in bank]) for key in FIELDS})


def load_bank(path):
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(FIELDS):
            raise ValueError('unexpected trace fields')
        # NPZ fields decompress on access. Materialize each field once, rather
        # than decompressing an entire bank for every episode.
        values = {key: arrays[key] for key in FIELDS}
        return [{key: values[key][e].copy() for key in FIELDS} for e in range(len(values['ambient']))]


def bank_identity(bank):
    episodes, arrays = [], {}
    for key in sorted(FIELDS):
        value = np.ascontiguousarray(np.stack([row[key] for row in bank]), dtype='<f8')
        arrays[key] = hashlib.sha256(str(value.shape).encode()+value.tobytes()).hexdigest()
    for row in bank:
        digest = hashlib.sha256()
        for key in sorted(FIELDS):
            value = np.ascontiguousarray(row[key], dtype='<f8')
            digest.update(key.encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
        episodes.append(digest.hexdigest())
    return dict(arrays=arrays, episodes=episodes,
                canonical_sha256=hashlib.sha256(''.join(episodes).encode()).hexdigest())


def reject_overlap(banks):
    """Compare canonical complete episodes and field arrays across named banks."""
    identities = {name: bank_identity(bank) for name, bank in banks.items()}
    names = list(identities)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            a, b = identities[name], identities[other]
            if set(a['episodes']) & set(b['episodes']):
                raise ValueError('duplicate complete episodes: '+name+' / '+other)
            if a['arrays'] == b['arrays']:
                raise ValueError('duplicate canonical arrays: '+name+' / '+other)
    return identities
