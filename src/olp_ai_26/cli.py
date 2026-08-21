"""Command-line entry points for inspection, submission validation, and notebook export."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Sequence
from pathlib import Path

from olp_ai_26.core.inspect import dataset_report, environment_report, load_table, save_report
from olp_ai_26.core.submission import validate_submission
from olp_ai_26.notebooks import build_all_notebooks


def _doctor(_: argparse.Namespace) -> int:
    report = environment_report()
    packages = ["numpy", "pandas", "scikit-learn", "torch", "torchvision", "timm", "transformers"]
    report["packages"] = {}
    for package in packages:
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    print(json.dumps(report, indent=2))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    report = dataset_report(args.data_dir, image_limit=args.image_limit)
    destination = save_report(report, args.output)
    print(f"Wrote {destination}")
    print(json.dumps(report["counts"], indent=2))
    return 0


def _validate(args: argparse.Namespace) -> int:
    submission = load_table(args.submission)
    sample = load_table(args.sample)
    result = validate_submission(submission, sample, id_columns=args.id_column)
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print("Submission is valid")
    return 0


def _build_notebooks(args: argparse.Namespace) -> int:
    outputs = build_all_notebooks(args.directory)
    for path in outputs:
        print(f"Wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``olp-ai`` argument parser and its competition utility commands."""
    parser = argparse.ArgumentParser(prog="olp-ai", description="OlympicAI competition toolkit")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="Show the local ML environment")
    doctor.set_defaults(handler=_doctor)

    inspect = commands.add_parser("inspect", help="Profile a competition dataset")
    inspect.add_argument("data_dir", type=Path)
    inspect.add_argument("--output", type=Path, default=Path("outputs/dataset_report.json"))
    inspect.add_argument("--image-limit", type=int, default=1000)
    inspect.set_defaults(handler=_inspect)

    validate = commands.add_parser("validate-submission", help="Check a CSV against the sample")
    validate.add_argument("submission", type=Path)
    validate.add_argument("sample", type=Path)
    validate.add_argument("--id-column", action="append", default=[])
    validate.set_defaults(handler=_validate)

    notebooks = commands.add_parser("build-notebooks", help="Export # %% templates to ipynb")
    notebooks.add_argument("--directory", type=Path, default=Path("notebooks"))
    notebooks.set_defaults(handler=_build_notebooks)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments and return the selected command's exit status."""
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
