import os
from pathlib import Path

WANDB_ORG_NAME = os.getenv("WANDB_ORG_NAME", "fekelemen-usp-org")
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME", "curriculum-learning")
OUTPUT_DIR_BASE: Path = Path("outputs/")
