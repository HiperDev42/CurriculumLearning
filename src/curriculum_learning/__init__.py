import argparse
from pathlib import Path

from dotenv import load_dotenv

from curriculum_learning.train import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to a .env file to load before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)

    configs: list[TrainingConfig] = [
        TrainingConfig(
            name="example_run",
            group="example_group",
            output_dir=Path("output/example_run"),
            model_base_name="bert-base-uncased",
            num_train_epochs=5,
        ),
        TrainingConfig(
            name="example_run_2",
            group="example_group",
            output_dir=Path("output/example_run_2"),
            model_base_name="bert-base-uncased",
            num_train_epochs=5,
        ),
    ]

    for config in configs:
        run_training(config)
