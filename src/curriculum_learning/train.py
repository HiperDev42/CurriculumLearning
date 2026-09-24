import logging
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from curriculum_learning.config import WANDB_PROJECT_NAME
from curriculum_learning.curriculum import CurriculumTrainer
from curriculum_learning.data import (
    calculate_difficulty,
    heuristic_functions,
    load_base_dataset,
    tokenize_dataset,
)

if TYPE_CHECKING:
    from transformers import Trainer

logger = logging.getLogger(__name__)


@cache
def load_metric(name: str) -> Any:
    import evaluate

    return evaluate.load(name)


@dataclass
class TrainingConfig:
    name: str
    group: str
    output_dir: Path
    model_base_name: str

    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_train_epochs: int = 3
    seed: int = 42

    curriculum_learning: bool = False
    heuristic_fn: str = "word_count"
    complexity_metric: str = "pragmatic_deletion_bzip2"
    num_levels: int = 3


def run_training(config: TrainingConfig, report_wandb: bool = True) -> "Trainer":
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        PreTrainedTokenizerBase,
        Trainer,
        TrainingArguments,
    )

    import wandb

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training output will be saved to {config.output_dir}")

    logger.info("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_base_name,
        num_labels=2,
    )
    base_dataset = load_base_dataset()
    tokenizer = AutoTokenizer.from_pretrained(config.model_base_name)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError(
            f"Expected a PreTrainedTokenizerBase, got {type(tokenizer).__name__}"
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    base_train_dataset = base_dataset["train"]
    train_dataset = tokenize_dataset(base_train_dataset, tokenizer)
    validation_dataset = tokenize_dataset(base_dataset["validation"], tokenizer)
    # test_dataset = tokenize_dataset(base_dataset["test"], tokenizer)

    if report_wandb:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            name=config.name,
            group=config.group,
            config=asdict(config),
            resume="never",
        )

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=10,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=config.seed,
        report_to=["wandb"] if report_wandb else [],
        run_name=f"{config.group}/{config.name}",
    )
    logger.info("Training device: %s", training_args.device)
    if training_args.device.type != "cuda":
        logger.warning(
            "Training is using %s instead of a CUDA GPU.", training_args.device
        )

    trainer: Trainer

    if config.curriculum_learning:
        logger.info(
            "Calculating difficulty levels for curriculum learning using heuristic function '%s' and complexity metric '%s'",
            config.heuristic_fn,
            config.complexity_metric,
        )

        heuristic_fn = heuristic_functions[config.heuristic_fn]
        difficulty = calculate_difficulty(
            base_train_dataset.to_pandas(),
            heuristic_fn=heuristic_fn,
            metric_name=config.complexity_metric,
        )

        logger.info("Using curriculum learning for training")
        trainer = CurriculumTrainer(
            difficulty=difficulty,
            num_levels=config.num_levels,
            curriculum_seed=config.seed,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    else:
        logger.info("Using standard training")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    try:
        trainer.train()
        return trainer
    finally:
        if report_wandb:
            wandb.finish()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logger.debug("Computing evaluation metrics for %d predictions", len(labels))
    predictions = np.argmax(logits, axis=-1)

    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(shifted_logits)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)

    accuracy_result = load_metric("accuracy").compute(
        predictions=predictions,
        references=labels,
    )
    auc_result = load_metric("roc_auc").compute(
        prediction_scores=probabilities[:, 1],
        references=labels,
    )

    if accuracy_result is None or auc_result is None:
        raise RuntimeError("Metrics were not computed on this process")

    metrics = {
        "accuracy": accuracy_result["accuracy"],
        "roc_auc": auc_result["roc_auc"],
    }
    logger.debug("Computed evaluation metrics: %s", metrics)
    return metrics
