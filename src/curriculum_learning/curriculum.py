import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
from transformers import Trainer

logger = logging.getLogger(__name__)


class CurriculumSampler(Sampler):
    """Sampler that restricts training to progressively harder difficulty levels.

    ``difficulty`` is sorted and cut into ``num_levels`` equal-size chunks,
    level 0 being the easiest. Rows are ordered by (score, original index)
    before cutting, so ties are broken by id order rather than being kept
    together, guaranteeing every level gets the same number of rows (up to
    a difference of one when the row count doesn't divide evenly).

    ``num_epochs`` (the total training epoch count) is split across the
    ``num_levels`` stages as evenly as possible; any remainder epochs go to
    the later (harder) stages. The active stage advances once per epoch via
    ``set_epoch`` (called automatically by ``Trainer``).

    The Trainer computes its per-epoch step count once, from the first epoch's
    dataloader length, and reuses that count for every subsequent epoch. So
    each epoch must yield the same number of indices; only *which* rows those
    indices point to may change. Rows visible at the current epoch's stage
    (``difficulty_level <= stage``) are drawn with replacement up to the full
    dataset length whenever that visible pool is smaller than the dataset.
    """

    def __init__(
        self,
        difficulty: pd.Series,
        num_levels: int,
        num_epochs: int,
        seed: int = 42,
        higher_is_harder: bool = True,
    ):
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if num_epochs < num_levels:
            raise ValueError(
                f"num_epochs ({num_epochs}) must be >= num_levels ({num_levels}) "
                "so every stage gets at least one epoch"
            )

        self.difficulty_levels = self._assign_levels(
            difficulty, num_levels, higher_is_harder
        )
        self.num_levels = num_levels
        stage_epoch_counts = self._build_stage_epoch_counts(num_epochs, num_levels)
        self.stage_boundaries = np.cumsum(stage_epoch_counts)
        self.seed = seed
        self.epoch = 0

    @staticmethod
    def _assign_levels(
        difficulty: pd.Series, num_levels: int, higher_is_harder: bool
    ) -> np.ndarray:
        if num_levels > len(difficulty):
            raise ValueError(
                f"num_levels ({num_levels}) must be <= number of rows "
                f"({len(difficulty)})"
            )

        scores = difficulty.to_numpy(dtype=float)
        if not higher_is_harder:
            scores = -scores

        # Stable sort keeps tied scores in original (id) order instead of
        # grouping them together, so the equal-size cut below is deterministic.
        order = np.argsort(scores, kind="stable")

        levels = np.empty(len(scores), dtype=np.int64)
        for level, chunk in enumerate(np.array_split(order, num_levels)):
            levels[chunk] = level
        return levels

    @staticmethod
    def _build_stage_epoch_counts(num_epochs: int, num_levels: int) -> list[int]:
        base, remainder = divmod(num_epochs, num_levels)
        counts = [base] * num_levels
        # Distribute leftover epochs to the later (harder) stages first.
        for i in range(remainder):
            counts[num_levels - 1 - i] += 1
        return counts

    def set_epoch(self, epoch: int) -> None:
        stage = self.current_stage(epoch)
        if epoch == 0 or stage != self.current_stage(epoch - 1):
            logger.info(
                "Curriculum sampler entering stage=%d at epoch=%d", stage, epoch
            )
        self.epoch = epoch

    def current_stage(self, epoch: int | None = None) -> int:
        """Get the current stage for the given epoch.

        Args:
            epoch (int | None, optional): The epoch for which to determine the current stage. Defaults to None, which uses the internally stored epoch.

        Returns:
            int: The current stage index corresponding to the given epoch.
        """
        epoch = self.epoch if epoch is None else epoch
        stage = int(np.searchsorted(self.stage_boundaries, epoch, side="right"))
        return min(stage, len(self.stage_boundaries) - 1)

    def __iter__(self):
        visible_indices = np.flatnonzero(self.difficulty_levels <= self.current_stage())
        num_samples = len(self.difficulty_levels)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)

        if len(visible_indices) >= num_samples:
            order = torch.randperm(len(visible_indices), generator=generator)[
                :num_samples
            ]
        else:
            order = torch.randint(
                0, len(visible_indices), (num_samples,), generator=generator
            )

        yield from visible_indices[order.numpy()].tolist()

    def __len__(self) -> int:
        return len(self.difficulty_levels)


class CurriculumTrainer(Trainer):
    """Trainer that builds a ``CurriculumSampler`` from its own epoch count.

    ``num_train_epochs`` already lives on ``TrainingArguments``, so the
    trainer reads it there instead of the caller passing it (or a derived
    per-stage split) separately.
    """

    def __init__(
        self,
        *args,
        difficulty: pd.Series,
        num_levels: int,
        curriculum_seed: int = 42,
        higher_is_harder: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        num_epochs = self.args.num_train_epochs
        if num_epochs != int(num_epochs):
            raise ValueError(
                f"num_train_epochs must be a whole number for curriculum "
                f"staging, got {num_epochs}"
            )

        self.curriculum_sampler = CurriculumSampler(
            difficulty=difficulty,
            num_levels=num_levels,
            num_epochs=int(num_epochs),
            seed=curriculum_seed,
            higher_is_harder=higher_is_harder,
        )

    def _get_train_sampler(self, train_dataset=None) -> torch.utils.data.Sampler | None:
        return self.curriculum_sampler
