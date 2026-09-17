import logging

import numpy as np
import torch
from torch.utils.data import Sampler
from transformers import Trainer

logger = logging.getLogger(__name__)


class CurriculumSampler(Sampler):
    """Sampler that restricts training to progressively harder difficulty levels.

    The Trainer computes its per-epoch step count once, from the first epoch's
    dataloader length, and reuses that count for every subsequent epoch. So
    each epoch must yield the same number of indices; only *which* rows those
    indices point to may change. Rows visible at the current epoch's stage
    (``difficulty_level <= stage``) are drawn with replacement up to the full
    dataset length whenever that visible pool is smaller than the dataset.

    The active stage advances once per epoch via ``set_epoch`` (called
    automatically by ``Trainer``), mapping epoch index to stage index using
    ``stage_epoch_counts`` (see ``build_stage_epoch_counts``).
    """

    def __init__(
        self,
        difficulty_levels: list[int],
        stage_epoch_counts: list[int],
        seed: int = 42,
    ):
        self.difficulty_levels = np.asarray(difficulty_levels)
        self.stage_boundaries = np.cumsum(stage_epoch_counts)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        stage = self.current_stage(epoch)
        if epoch == 0 or stage != self.current_stage(epoch - 1):
            logger.info(
                "Curriculum sampler entering stage=%d at epoch=%d", stage, epoch
            )
        self.epoch = epoch

    def current_stage(self, epoch: int | None = None) -> int:
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
    """Trainer that draws training batches from a ``CurriculumSampler``."""

    def __init__(self, *args, curriculum_sampler: CurriculumSampler, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_sampler = curriculum_sampler

    def _get_train_sampler(self, train_dataset=None) -> torch.utils.data.Sampler | None:
        return self.curriculum_sampler
