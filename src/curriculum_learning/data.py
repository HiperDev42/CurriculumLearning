import hashlib
import json
import logging
import random
import tempfile
from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from lang_complexity.complexity import complexities
from transformers import PreTrainedTokenizerBase

BASE_DATASET_NAME = "nilc-nlp/assin2"


@lru_cache(maxsize=1)
def load_base_dataset(revision: str | None = None) -> DatasetDict:
    return load_dataset(BASE_DATASET_NAME, revision=revision)


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
    *,
    heuristic_name: str | None = None,
    dataset_fingerprint: str | None = None,
    cache_dir: Path | None = None,
    seed: int | None = None,
) -> pd.Series:
    df = dataframe
    group_size = 250

    if metric_name not in complexities:
        raise ValueError(f"Metric '{metric_name}' is not available in complexities.")
    metric_fn = complexities[metric_name].compute

    if df.empty:
        return pd.Series(dtype=float, name="difficulty")

    if cache_dir is not None and (
        heuristic_name is None or dataset_fingerprint is None
    ):
        raise ValueError(
            "heuristic_name and dataset_fingerprint are required when caching difficulty"
        )
    if cache_dir is not None and heuristic_name == "random" and seed is None:
        raise ValueError("A seed is required to cache the random heuristic")

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_key_data = {
            "schema": 1,
            "dataset_fingerprint": dataset_fingerprint,
            "heuristic": heuristic_name,
            "metric": metric_name,
            "metric_version": version("lang-complexity"),
            "group_size": group_size,
            "seed": seed if heuristic_name == "random" else None,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_key_data, sort_keys=True).encode()
        ).hexdigest()
        cache_path = cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            cached_difficulty = np.load(cache_path, allow_pickle=False)
            if cached_difficulty.shape != (len(df),):
                raise ValueError(
                    f"Cached difficulty has shape {cached_difficulty.shape}, "
                    f"expected ({len(df)},): {cache_path}"
                )
            logger.info("Loaded cached difficulty scores from %s", cache_path)
            return pd.Series(
                cached_difficulty,
                index=df.index,
                dtype=float,
                name="difficulty",
            )

    if heuristic_name == "random" and seed is not None:
        rng = random.Random(seed)
        heuristic_score = df.apply(lambda _: rng.random(), axis=1)
    else:
        heuristic_score = df.apply(heuristic_fn, axis=1)
    sorted_index = heuristic_score.sort_values().index

    difficulty = pd.Series(index=df.index, dtype=float, name="difficulty")

    for start in range(0, len(sorted_index), group_size):
        group_index = sorted_index[start : start + group_size]
        group = df.loc[group_index]
        text = "\n".join(group["premise"].tolist() + group["hypothesis"].tolist())
        difficulty.loc[group_index] = metric_fn(text)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=cache_path.parent,
                prefix=f".{cache_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as cache_file:
                temporary_path = Path(cache_file.name)
                np.save(cache_file, difficulty.to_numpy(dtype=np.float64))
            temporary_path.replace(cache_path)
            logger.info("Saved difficulty scores to cache %s", cache_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return difficulty


def tokenize_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase):
    logger.debug("Tokenizing dataset with %d rows", len(dataset))

    def tokenize_fn(batch):
        return tokenizer(
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
