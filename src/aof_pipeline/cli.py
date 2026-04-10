from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from aof_pipeline.pipeline import create_demo_dataset, resolve_path, run_pipeline


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cmd_demo_data(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    generated = create_demo_dataset(output_dir=output_dir, count=args.count, seed=args.seed)
    print(json.dumps({"generated": len(generated), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    summary = run_pipeline(config=config, config_dir=config_path.parent)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AoF-200K data construction pipeline reproduction.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo-data", help="Generate synthetic images for pipeline debugging.")
    demo_parser.add_argument("--output-dir", required=True, help="Directory for generated demo images.")
    demo_parser.add_argument("--count", type=int, default=12, help="Number of demo images.")
    demo_parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    demo_parser.set_defaults(func=cmd_demo_data)

    run_parser = subparsers.add_parser("run", help="Run the full data construction pipeline.")
    run_parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    run_parser.set_defaults(func=cmd_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
