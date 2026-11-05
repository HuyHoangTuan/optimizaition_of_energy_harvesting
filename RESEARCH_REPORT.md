# Independent learning: progress and limits

## Section 1: Why

Controlled independent DQN learned across the tested transmitter counts. Corrected one-SU DRQN did not establish a consistent improvement.

The research asks whether secondary users (SUs), simulated secondary radio transmitters, can learn their own energy-harvesting policies. It also asks whether memory helps with delayed observations. A shared controller broadcasts one action and learns summed reward. Independent agents choose together and learn only their own reward.

Historical source recovery showed qualitative learning, but the recovered candidate is not proven to have generated the original logs. Exact paper or saved-curve reproduction is unestablished. Different source models, actions, debit rules and banks prevent one causal leaderboard. Reward is not validated physical throughput.

## Section 2: Summary

- Baseline recovery beat matched random controls, with variable dynamic generalization.
- The early independent DQN recipe failed at N = 1, 3, 5 despite working agent isolation.
- Controlled DQN passed recorded study gates at fixed N = 1, 2, 3, 5 and dynamic N = 1, 3.
- Corrected fixed N = 1 DRQN gave mean paired gain +0.010206447; one of three seeds improved. Its frozen criterion failed.
- Multi-SU DRQN is implemented, but N = 2 was not admitted by budget and N = 3 was skipped.

Episodes contain 20 slots. Baseline results use reward per episode; later results use reward per SU per slot. Reward includes penalties; recorded score counts executed contributions. Fixed rho is 0.4; the dynamic set {1/8,…,7/8} excludes 0.4. Three training seeds remain three training replicates, even with 100 evaluation episodes per policy.

The [standalone report](RESEARCH_REPORT.html) embeds the checked comparison charts, neural diagrams, radio diagram and equations.

## Section 3: Changes

This PR includes the independent-agent and stability implementation, plus corrected DRQN and public documentation. Each SU owns policy, target, optimizer, replay, random generators and, for DRQN, memory.

The early evaluated recipe used 64 × 64 layers, Adam 0.001, Huber loss, batch 8 and executed-only debit. Controlled DQN uses 128 × 128 layers, SGD 0.003, MSE, batch 64, gamma 0.99 and requested debit. Targets update with tau 0.001 every fifth zero-based episode, including zero. Double DQN is supported, but the selected recipe uses plain target maximization and no clipping. The studies changed several factors; they do not identify a single cause.

Corrected [GRU64 replay](src/multi_agent/recurrent.py) stores 20-transition episodes with 21 observations. Policy and target process four full episodes from zero. Successor histories retain observation zero; 64 of 80 positions enter MSE. Acting memory advances during random actions, refreshes after updates, and resets between episodes. The legacy recurrent path did not satisfy this chronological contract.

The local TD target is `y = r + 0.99 * (1-d) * max Q_target(next_history)`. Terminal targets contain reward only. Corrected one-SU DQN/DRQN use 22,048/16,672 parameters, so the comparison does not isolate memory alone.

The [environment](src/multi_agent/environment.py) applies previous harvest and debit at the next battery boundary. Requested debit can charge rejected transmit requests. Both action branches harvest. Numeric dBm score arithmetic, density-grid channels and per-SU primary checks do not validate physical throughput or aggregate primary protection. Historical equations differ from this later model.

See [MULTI_AGENT.md](MULTI_AGENT.md) for independent DQN and [DRQN.md](DRQN.md) for recurrent checks and external study-input requirements.

## Section 4: Result

Recovered baseline held-out reward per 20-slot episode:

| Source / mode | Training seeds | Final | Random |
| --- | --- | --- | --- |
| Historical fixed | 3 | 27.960459 | −6.220458 |
| Historical dynamic | 3 | 13.940750 | −7.109862 |
| Current-source fixed | 1 | 15.567203 | −6.525259 |
| Current-source dynamic | 1 | 17.329592 | −6.723132 |

Historical rows use seeds 2, 17, 42; current controls use seed 2. Saved late-training means 26.473710/42.831108 are a different evidence class. Historical dynamic held-out seeds were 33.626409, 9.840545 and −1.644706; all beat random, but one remained negative.

Early independent recipe, three seeds per group, reward per SU per slot:

| N | Final | Random | Final − random |
| --- | --- | --- | --- |
| 1 | −0.420125 | −0.218698 | −0.201427 |
| 3 | −0.763595 | −0.291785 | −0.471810 |
| 5 | −0.411412 | −0.338165 | −0.073247 |

Twenty of 27 trained SU policies never executed successfully. Excessive action-value estimates accompanied the failure; these observations do not identify one proven cause.

Controlled DQN, three seeds per group, reward per SU per slot:

| Mode / N | Final | Random | Execution |
| --- | --- | --- | --- |
| Fixed / 1 | 0.851535 | −0.323089 | 94.5167% |
| Fixed / 2 | 0.498276 | −0.349034 | 94.4167% |
| Fixed / 3 | 0.360747 | −0.351447 | 93.3167% |
| Fixed / 5 | 0.228109 | −0.374923 | 92.2367% |
| Dynamic / 1 | 0.936180 | −0.302057 | 94.7167% |
| Dynamic / 3 | 0.365883 | −0.350478 | 93.6889% |

All six groups beat initial and random controls and passed recorded learning, health, score and execution gates. Per-SU reward declines with N; network reward per slot rises from 0.851535 at fixed N = 1 to approximately 1.140545 at N = 5. Gates do not prove convergence or optimality. A one-seed shared-controller pilot cannot establish superiority because observations, objectives, capacity and actions differ.

Corrected fixed N = 1, final episode-1600 checkpoints, reward per SU per slot:

| Training seed | DQN | DRQN | DRQN − DQN |
| --- | --- | --- | --- |
| 2 | 0.860216 | 0.854832 | −0.005384 |
| 17 | 0.753696 | 0.818894 | +0.065197 |
| 42 | 0.835579 | 0.806386 | −0.029194 |
| Mean | 0.816497 | 0.826704 | +0.010206 |

The full-precision mean paired gain is +0.010206447, sample SD 0.049089. The criterion required mean gain at least 0.05 and at least two positive seeds; neither passed. Rounded values can differ from subtraction of displayed means. All six final policies beat initial and random controls. Mean score was 0.835831 DQN versus 0.834370 DRQN, so reward gain was not score gain.

Paired banks, action maps and checkpoint timing were matched; capacity, replay, warmup, history, RNG consumption and compute were not. Use the newly trained paired DQN here, rather than the earlier stability mean. DRQN charged 644.356997 versus 209.533602 training wall seconds across three cells, approximately 3.08 times as much, excluding overhead. This is not inference speed or energy efficiency.

The study records 37 scientific tests and independent scientific model review. A fresh public-checkout rerun also passed all 37 tests with exit 0. Document checks do not establish independent full-report browser acceptance or human acceptance. No new experiment ran for this report.

Full frozen-study reproduction needs the exact external protocol and prior trace bundle; these are not included. The next research step is a defined memory hypothesis, paired baselines, more seeds and an agreed budget. Multi-SU recurrent performance remains untested.
