import logging
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from curriculum_learning.config import PROJECT_NAME
from curriculum_learning.curriculum import CurriculumSampler, CurriculumTrainer
from curriculum_learning.data import load_base_dataset, tokenize_dataset

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


def run_training(config: TrainingConfig):
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
    train_dataset = tokenize_dataset(base_dataset["train"], tokenizer)
    validation_dataset = tokenize_dataset(base_dataset["validation"], tokenizer)
    # test_dataset = tokenize_dataset(base_dataset["test"], tokenizer)

    wandb.init(
        project=PROJECT_NAME,
        name=config.name,
        group=config.group,
        config=asdict(config),
        resume=True,
    )

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        seed=config.seed,
        report_to=["wandb"],
        run_name=f"{config.group}/{config.name}",
    )

    trainer: Trainer

    if config.curriculum_learning:
        trainer = CurriculumTrainer(
            curriculum_sampler=CurriculumSampler(
                difficulty_levels=[0] * len(train_dataset),
                stage_epoch_counts=[config.num_train_epochs],
                seed=config.seed,
            ),
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    return trainer.train()


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
