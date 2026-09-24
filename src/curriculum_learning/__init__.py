import argparse
from pathlib import Path
from typing import Any, cast

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to a .env file to load before training.",
    )
    args, _ = parser.parse_known_args()
    return args


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    env_file = args.env_file or project_root / ".env"
    load_dotenv(env_file)

    from curriculum_learning.train import TrainingConfig, run_training

    values = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))

    values["output_dir"] = Path(values["output_dir"])
    training_config = TrainingConfig(**values)

    run_training(training_config)


if __name__ == "__main__":
    main()
