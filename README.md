# optimizaition_of_energy_harvesting

# 1. Introduction

# 2. Installation

## 2.1. Requirements

## 2.2. Installation

# 3. Documents

## 3.1. Slides
https://www.canva.com/design/DAF-JdEtyEw/_lCoKLk3Q2EtHJrTODqrhw/edit?utm_content=DAF-JdEtyEw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## 3.2.

## Independent agents per secondary transmitter

The separate CPU path `python train_multi_agent.py --num-su 3 --episodes 2 --slots 20`
creates one independent DQN per SU, trained with that SU's own reward. See
[MULTI_AGENT.md](MULTI_AGENT.md) for installation, the explicit compatibility-score
model, observation/battery timing, artifacts and focused tests. Legacy entry points
and historical results remain available.

## Research results and corrected recurrence

Controlled independent DQN learned across the tested SU counts. Corrected one-SU DRQN produced mixed results and failed its improvement criterion; no multi-SU DRQN comparison completed.

Read the four-section [research report](RESEARCH_REPORT.md) or the [standalone HTML report](RESEARCH_REPORT.html). Use [DRQN.md](DRQN.md) for local checks and exact external frozen-protocol/prior-input requirements. The checkout supports tests; it does not include all inputs for full frozen-study reproduction.
