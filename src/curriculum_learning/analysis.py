import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    predictions: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray


def get_best_checkpoint(model_dir: Path) -> Path:
    state_paths = [
        model_dir / "trainer_state.json",
        *model_dir.glob("checkpoint-*/trainer_state.json"),
    ]
    for state_path in state_paths:
        if not state_path.exists():
            continue
        with state_path.open() as state_file:
            best_checkpoint = json.load(state_file).get("best_model_checkpoint")
        if best_checkpoint:
            checkpoint_dir = Path(best_checkpoint)
            if checkpoint_dir.exists():
                return checkpoint_dir
            local_checkpoint_dir = model_dir / checkpoint_dir.name
            if local_checkpoint_dir.exists():
                return local_checkpoint_dir
    raise FileNotFoundError(f"No best checkpoint found in {model_dir}")


def predict_pair(
    model: BertForSequenceClassification,
    tokenizer: PreTrainedTokenizerBase,
    premise: str,
    hypothesis: str,
) -> tuple[int, list[float]]:
    """Predict the entailment label and class probabilities for a single premise/hypothesis pair."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: BertForSequenceClassification = model.to(device)  # type: ignore
    model.eval()

    inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    inputs = {name: values.to(device) for name, values in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0]
    predicted_label = int(probabilities.argmax())
    return predicted_label, probabilities.tolist()


def predict(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> PredictionResult:
    """Run a classification model over a tokenized dataset with 'labels' column.

    The dataset must already be tokenized (input_ids/attention_mask present).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # type: ignore
    model.eval()

    dataset = dataset.with_format("torch")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    predictions: list[int] = []
    probabilities: list[float] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch_labels = batch.pop("labels")
            inputs = {name: values.to(device) for name, values in batch.items()}
            logits = model(**inputs).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            labels.extend(batch_labels.tolist())

    return PredictionResult(
        predictions=np.asarray(predictions),
        probabilities=np.asarray(probabilities),
        labels=np.asarray(labels),
    )


def compute_prediction_metrics(result: PredictionResult) -> dict[str, float]:
    """Compute accuracy and ROC AUC from a PredictionResult."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    return {
        "accuracy": accuracy_score(result.labels, result.predictions),
        "roc_auc": roc_auc_score(result.labels, result.probabilities),
    }
