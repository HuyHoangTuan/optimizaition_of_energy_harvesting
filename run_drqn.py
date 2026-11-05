"""Explicit, portable entry point for the separately reviewed DRQN study."""
import argparse
from pathlib import Path
import json

from src.multi_agent import drqn_experiment as experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('preflight','pilot','study','audit','report','verify'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--legacy-root',type=Path)
    parser.add_argument('--legacy-manifest',type=Path)
    parser.add_argument('--freeze',action='store_true',help='Freeze only after independent pretraining verification')
    args=parser.parse_args()
    experiment.configure_runtime()
    if args.stage=='preflight':
        if not all((args.protocol,args.legacy_root,args.legacy_manifest)):
            parser.error('preflight requires --protocol, --legacy-root and --legacy-manifest')
        result=experiment.preflight(args.output,args.protocol,args.legacy_root,args.legacy_manifest,do_freeze=args.freeze)
    elif args.stage=='report':
        from src.multi_agent.drqn_analysis import report
        result=report(args.output)
    else:
        if args.freeze:parser.error('--freeze is permitted only for preflight')
        result=getattr(experiment,args.stage)(args.output)
    print(json.dumps(result,allow_nan=False,indent=2))


if __name__=='__main__':main()
