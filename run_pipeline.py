"""
End-to-end pipeline runner.

Usage:
    python run_pipeline.py                  # every stage
    python run_pipeline.py --stage train    # one stage
    python run_pipeline.py --stage decide

Stages run in this order and each reads what the earlier ones wrote to outputs/:
    sql, train, decide, experiment, explain, report
"""

import argparse
import warnings

from src.pipeline import STAGES, run_all, run_stage

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Customer churn analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["all"] + STAGES,
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    if args.stage == "all":
        run_all()
    else:
        run_stage(args.stage)

    print("\nDone. Results are in outputs/ (figures in outputs/figures/).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
