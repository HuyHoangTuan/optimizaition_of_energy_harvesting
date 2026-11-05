"""Amendable report-only analysis. Excluded from the scientific source freeze."""
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np


def report(root):
    root=Path(root)
    # This module reads results and never imports or runs training functions.
    audit=root/'audit/summary.json'
    if not audit.exists():raise RuntimeError('completed audit summary required for results report')
    values=json.loads(audit.read_text())
    comparisons=[]
    for n in (1,2,3):
        pairs=[]
        for seed in (2,17,42):
            dqn=values.get(f'cell_dqn_n{n}_seed{seed}')
            drqn=values.get(f'cell_drqn_n{n}_seed{seed}')
            if dqn is not None and drqn is not None:
                pairs.append(dict(seed=seed,dqn=dqn['reward_per_su_slot'],drqn=drqn['reward_per_su_slot'],
                                  difference=drqn['reward_per_su_slot']-dqn['reward_per_su_slot']))
        if pairs:
            diff=np.array([p['difference'] for p in pairs])
            comparisons.append(dict(num_su=n,pairs=pairs,mean=float(diff.mean()),
                sample_sd=float(diff.std(ddof=1)) if len(diff)>1 else None,
                complete_three_seed_comparison=len(pairs)==3,
                descriptive_improvement=bool(len(pairs)==3 and diff.mean()>=.05 and sum(diff>0)>=2)))
    folder=root/'reports'/uuid.uuid4().hex;folder.mkdir(parents=True,exist_ok=False)
    provenance=dict(analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    audit_summary_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
                    seal_sha256=hashlib.sha256((root/'seal.json').read_bytes()).hexdigest(),
                    analysis_only=True)
    (folder/'comparison.json').write_text(json.dumps(dict(provenance=provenance,comparisons=comparisons),indent=2))
    lines=['# Corrected DRQN comparison','',
           'These results describe the frozen synthetic requested-debit compatibility model.','',
           'Three training seeds support descriptive comparisons. They do not establish statistical superiority.','',
           '| SU count | Paired differences (DRQN minus DQN) | Mean | Sample SD |',
           '|---|---|---:|---:|']
    for row in comparisons:
        sd='pending' if row['sample_sd'] is None else f"{row['sample_sd']:.6f}"
        lines.append(f"| {row['num_su']} | {', '.join(format(p['difference'],'.6f') for p in row['pairs'])} | {row['mean']:.6f} | {sd} |")
    lines.extend(['','The initial, random and final audit episode and slot records remain in the audit archive.',
                  'A missing or incomplete stage is a budget, gate or failure outcome; it is not evidence of architecture inferiority.',
                  'Parameter counts, replay grouping, history, RNG consumption, warmup updates and runtime differ.',
                  'No physical validity, paper numerical reproduction, universal superiority or professor acceptance is established.'])
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    if comparisons:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        for row in comparisons:
            ax.scatter([row['num_su']]*len(row['pairs']),[p['difference'] for p in row['pairs']],label=f"{row['num_su']} SU")
            ax.plot(row['num_su'],row['mean'],marker='_',color='black',markersize=22)
        ax.axhline(0,color='gray',linewidth=1);ax.set_xlabel('Secondary users')
        ax.set_ylabel('Audit reward / SU / slot: DRQN − DQN')
        fig.tight_layout();fig.savefig(folder/'paired-differences.png',dpi=160)
        fig.savefig(folder/'paired-differences.svg');plt.close(fig)
    return dict(status='report_created',directory=str(folder),comparisons=comparisons)
