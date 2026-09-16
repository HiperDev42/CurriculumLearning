import logging

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

BASE_DATASET_NAME = "nilc-nlp/assin2"


def load_base_dataset():
    return load_dataset(BASE_DATASET_NAME)


logger = logging.getLogger(__name__)


def tokenize_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase):
    logger.debug("Tokenizing dataset with %d rows", len(dataset))
    tokenize_fn = lambda batch: tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=256,
    )

    processed_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[
            "sentence_pair_id",
            "premise",
            "hypothesis",
            "relatedness_score",
        ],
    ).rename_column("entailment_judgment", "labels")
    logger.debug("Finished tokenizing dataset with %d rows", len(processed_dataset))
    return processed_dataset
