import logging
import random
from collections.abc import Callable
from functools import lru_cache

import pandas as pd
from datasets import Dataset, load_dataset
from lang_complexity.complexity import complexities
from transformers import PreTrainedTokenizerBase

BASE_DATASET_NAME = "nilc-nlp/assin2"


@lru_cache(maxsize=1)
def load_base_dataset():
    return load_dataset(BASE_DATASET_NAME)


logger = logging.getLogger(__name__)


heuristic_functions = {
    "random": lambda row: random.random(),
    "word_count": lambda row: (
        len(row["premise"].split()) + len(row["hypothesis"].split())
    ),
    "char_count": lambda row: len(row["premise"]) + len(row["hypothesis"]),
    "lexical_diversity": lambda row: (
        len(set(row["premise"].split() + row["hypothesis"].split()))
        / (len(row["premise"].split()) + len(row["hypothesis"].split()))
    ),
}


def calculate_difficulty(
    dataframe: pd.DataFrame,
    heuristic_fn: Callable = heuristic_functions["word_count"],
    metric_name: str = "pragmatic_deletion_bzip2",
) -> pd.Series:
    df = dataframe
    group_size = 250

    if metric_name not in complexities:
        raise ValueError(f"Metric '{metric_name}' is not available in complexities.")
    metric_fn = complexities[metric_name].compute

    if df.empty:
        return pd.Series(dtype=float, name="difficulty")

    heuristic_score = df.apply(heuristic_fn, axis=1)
    sorted_index = heuristic_score.sort_values().index

    difficulty = pd.Series(index=df.index, dtype=float, name="difficulty")

    for start in range(0, len(sorted_index), group_size):
        group_index = sorted_index[start : start + group_size]
        group = df.loc[group_index]
        text = "\n".join(group["premise"].tolist() + group["hypothesis"].tolist())
        difficulty.loc[group_index] = metric_fn(text)

    return difficulty


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
