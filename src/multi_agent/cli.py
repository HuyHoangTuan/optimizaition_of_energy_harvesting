"""Bounded CPU entry point, independent of the legacy package tree."""
import argparse
import json

import torch

from .agent import AgentConfig
from .environment import EnvironmentConfig
from .trainer import MultiAgentTrainer


def main(argv=None):
    parser = argparse.ArgumentParser(description="One independent own-reward DQN per SU (compatibility scores)")
    parser.add_argument("--num-su", type=int, default=2)
    parser.add_argument("--slots", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--dynamic-rho", action="store_true")
    parser.add_argument("--preset", choices=("default", "legacy-parity"), default="default")
    parser.add_argument("--debit-mode", choices=("executed", "requested"), default="executed")
    parser.add_argument("--double-dqn", action="store_true")
    parser.add_argument("--output", help="Must not exist; default .cache/multi_agent/<unique-run>")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    torch.set_num_threads(1)
    try:
        overrides = {k: v for k, v in {"batch_size": args.batch_size, "hidden": args.hidden}.items() if v is not None}
        config = (AgentConfig.legacy_parity(**overrides, double_dqn=args.double_dqn)
                  if args.preset == "legacy-parity" else AgentConfig(**{**{"batch_size": 8}, **overrides}, double_dqn=args.double_dqn))
        trainer = MultiAgentTrainer(EnvironmentConfig(num_su=args.num_su, slots=args.slots, seed=args.seed,
                                                      dynamic_rho=args.dynamic_rho, debit_mode=args.debit_mode),
                                    config)
        output, summary = trainer.run(args.episodes, args.output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(output), **summary}, allow_nan=False))
    return 0
