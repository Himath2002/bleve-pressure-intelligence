"""Generate deterministic synthetic records for exercising the public workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from bleve_pressure.demo import generate_scenarios, synthetic_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--train-rows", type=int, default=600)
    parser.add_argument("--test-rows", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_rows < 100 or args.test_rows < 1:
        parser.error("--train-rows must be at least 100 and --test-rows must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = generate_scenarios(args.train_rows, seed=args.seed, first_id=1)
    train["Target Pressure (bar)"] = synthetic_target(train, seed=args.seed + 1)
    test = generate_scenarios(
        args.test_rows,
        seed=args.seed + 2,
        first_id=args.train_rows + 1,
    )

    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    print(f"Synthetic training records: {len(train):,}")
    print(f"Synthetic test records: {len(test):,}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
