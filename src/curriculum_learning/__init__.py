import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


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
    project_root = Path(__file__).resolve().parents[2]
    env_file = args.env_file or project_root / ".env"
    load_dotenv(env_file)

    from curriculum_learning.train import TrainingConfig, run_training

    run_group = datetime.now().astimezone().strftime("execution-%Y%m%d-%H%M%S-%z")
    model_base_name = "bert-base-uncased"
    num_train_epochs = 50

    configs: list[TrainingConfig] = [
        TrainingConfig(
            name="baseline",
            output_dir=Path("output/baseline"),
            curriculum_learning=False,
            group=run_group,
            model_base_name=model_base_name,
            num_train_epochs=num_train_epochs,
        ),
        TrainingConfig(
            name="curriculum_baby_steps",
            output_dir=Path("output/curriculum_baby_steps"),
            curriculum_learning=True,
            group=run_group,
            model_base_name=model_base_name,
            num_train_epochs=num_train_epochs,
        ),
    ]

    for config in configs:
        run_training(config)
