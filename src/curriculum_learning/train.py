import logging
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoModelForSequenceClassification

OUTPUT_DIR_BASE: Path = Path("output/")

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    output_dir: Path
    model_base_name: str


def run_training(config: TrainingConfig):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training output will be saved to {config.output_dir}")

    logger.info("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_base_name,
        num_labels=2,
    )
