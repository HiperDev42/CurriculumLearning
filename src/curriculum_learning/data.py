import logging
import random
from collections.abc import Callable
from functools import lru_cache

import numpy as np
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
    dataset: Dataset,
    heuristic_fn: Callable = heuristic_functions["word_count"],
    metric_name: str = "pragmatic_deletion_bzip2",
) -> Dataset:
    df = dataset.to_pandas()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected Dataset.to_pandas() to return a DataFrame")

    df["heuristic_score"] = df.apply(heuristic_fn, axis=1)
    df.sort_values(by="heuristic_score", inplace=True)

    groups = [df.iloc[i : i + 250] for i in range(0, len(df), 250)]
    metric_fn = complexities[metric_name].compute

    for group in groups:
        sentences = [row["premise"] for _, row in group.iterrows()] + [
            row["hypothesis"] for _, row in group.iterrows()
        ]
        text = "\n".join(sentences)
        group["complexity_score"] = metric_fn(text)

    df = pd.concat(groups)

    return Dataset.from_pandas(df)


def annotate_difficulty_levels(
    dataset: Dataset,
    heuristic_fn: Callable = heuristic_functions["word_count"],
    metric_name: str = "pragmatic_deletion_bzip2",
    num_difficulty_levels: int = 3,
) -> Dataset:
    if num_difficulty_levels < 1:
        raise ValueError("num_difficulty_levels must be at least 1")

    processed_dataset = calculate_difficulty(dataset, heuristic_fn, metric_name)

    df = processed_dataset.to_pandas()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected Dataset.to_pandas() to return a DataFrame")

    sorted_indices = df.sort_values(by="complexity_score").index
    df["difficulty_level"] = 0
    for level, group_indices in enumerate(
        np.array_split(sorted_indices, num_difficulty_levels)
    ):
        df.loc[group_indices, "difficulty_level"] = level

    return Dataset.from_pandas(df, preserve_index=False)


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
