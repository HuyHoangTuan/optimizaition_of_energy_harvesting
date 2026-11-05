# Corrected DRQN checks and study inputs

Run the local tests to check corrected recurrent behavior. Full frozen-study reproduction requires external inputs that this checkout does not include.

The completed fixed one-SU study gave mean paired gain +0.010206447 reward per SU per slot, with one positive seed of three. It failed the frozen improvement criterion. N = 2 was not admitted by budget; N = 3 was skipped. Read the [research report](RESEARCH_REPORT.md) or [standalone HTML report](RESEARCH_REPORT.html) for results and limits.

## Run local checks

Use Python 3.10 and an environment with the pinned [dependencies](requirements-multi-agent.txt). The observed study runtime used Python 3.10.20, NumPy 1.26.1 and PyTorch 2.1.0+cpu. The report-generation stage additionally uses Matplotlib 3.8.0, which is not pinned in that dependency file.

From the project root, run:

```text
python -m unittest discover -s tests -p "test*.py" -v
```

These deterministic checks do not train or reproduce the full study. Tests cover local scientific behavior; they do not establish physical validity.

## Prepare external study inputs

Obtain the exact frozen protocol and retained prior stability trace/source bundle from the experiment custodian. They are external dependencies, not downloadable files supplied by this repository. The expected protocol SHA256 is:

```text
95bb3fa433489fcf7e4bb3792264bb3c9495e8f95eb744d7fa7fc698c6686027
```

Use the required prior bundle with its matching `artifact_sha256.json`. A substitute protocol, newly generated traces or arbitrary directory will not reproduce the frozen study. The source/runtime freeze and exact input identities must also match. This checkout alone is not a self-contained experiment rerun.

After obtaining those inputs, use an unused output path for a non-training preflight. Replace each uppercase path below with the matching external input location:

```text
python run_drqn.py --stage preflight --output NEW_STUDY --protocol FROZEN_PROTOCOL.json --legacy-root PRIOR_STABILITY_BUNDLE --legacy-manifest PRIOR_STABILITY_BUNDLE/artifact_sha256.json
```

Preflight validates prior training banks and action maps and creates monitor/pilot inputs. It does not open audit banks. An existing output directory is refused. This example has not been executed from the public checkout with an exported bundle.

A full run additionally requires a verified source/runtime/input freeze, independent pretraining verification and the protocol's cost/admission controls. Do not infer training authorization from this quickstart. Inspect [the stage runner](run_drqn.py), [experiment controls](src/multi_agent/drqn_experiment.py), [trace checks](src/multi_agent/drqn_traces.py) and [evaluation](src/multi_agent/drqn_evaluation.py) before planning a separate run.

## Interpret the implementation

Each SU owns its GRU64, target, optimizer, episode replay, RNGs and acting state. Replay uses full chronological histories and terminal masking. Imports do no training. Checkpoints support evaluation; they omit replay/RNG state and cannot resume training exactly.

The synthetic requested-debit model retains numeric dBm score arithmetic and delayed observations. Model score is not validated physical throughput. See [independent-agent model details](MULTI_AGENT.md).
