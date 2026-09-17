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
            name="baseline",
            group="example_group",
            output_dir=Path("output/baseline"),
            model_base_name="bert-base-uncased",
            num_train_epochs=5,
        ),
        TrainingConfig(
            name="curriculum_baby_steps",
            group="example_group",
            output_dir=Path("output/curriculum_baby_steps"),
            model_base_name="bert-base-uncased",
            num_train_epochs=5,
            curriculum_learning=True,
        ),
    ]

    for config in configs:
        run_training(config)
