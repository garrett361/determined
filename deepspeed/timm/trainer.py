import logging
import random
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, Union

import determined as det
import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import tqdm
from determined.pytorch import TorchData

import data


class DeepSpeedTrainer(nn.Module):
    def __init__(
        self,
        core_context: det.core.Context,
        latest_checkpoint: str,
        model: nn.Module,
        transforms: Union[Callable, List[Callable]],
        ds_config: Dict[str, Any],
        train_batch_size: int,
        val_batch_size: int,
        dataset_name: str,
        sanity_check: bool = False,
        lr: Optional[float] = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.core_context = core_context
        self.latest_checkpoint = latest_checkpoint
        self.model = model
        self.transforms = transforms
        self.ds_config = ds_config
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataset_name = dataset_name
        self.sanity_check = sanity_check
        if self.sanity_check:
            logging.info(f"Running in sanity check mode!")
        self.lr = lr
        self.random_seed = random_seed

        self.criterion = nn.CrossEntropyLoss()

        self.rank = core_context.distributed.rank
        self.is_distributed = core_context.distributed.size > 1
        self.is_chief = self.rank == 0
        self.device = f"cuda:{self.rank}"

        self.steps_completed = 0

        # Instantiated as needed through private methods.
        self.train_dataset = None
        self.train_loader = None
        self.val_dataset = None
        self.val_loader = None
        self.model_engine = None
        self.optimizer = None
        self.fp16 = None

        self._setup()

    def _set_random_seeds(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.random.manual_seed(self.random_seed)

    def _build_datasets(self) -> None:
        self.train_dataset = data.get_dataset(
            dataset_name=self.dataset_name, split="train", transforms=self.transforms
        )
        self.val_dataset = data.get_dataset(
            dataset_name=self.dataset_name, split="val", transforms=self.transforms
        )

    def _build_metrics(self) -> None:
        # Create metrics for the various splits.  These all take in outputs, targets pairs as args
        # where outputs are either probabilities or single predictions.
        self.metrics = {"train": {}, "val": {}, "test": {}}
        # Metrics for all
        for split, met_dic in self.metrics.items():
            for k in range(1, 11):
                met_dic[f"{split}_top{k}_acc"] = torchmetrics.Accuracy(top_k=k)
        for vals in self.metrics.values():
            for metric in vals.values():
                if metric is not None:
                    metric.to(self.device)

    def _deepspeed_init(self) -> None:
        deepspeed.init_distributed()
        self.model_engine, self.optimizer, self.train_loader, __ = deepspeed.initialize(
            model=self.model,
            training_data=self.train_dataset,
            config=self.ds_config,
        )
        self.fp16 = self.model_engine.fp16_enabled()

    def _setup(self) -> None:
        self._build_datasets()
        self._build_val_loader()
        self._build_test_loader()
        self._build_metrics()
        self._deepspeed_init()

    def _batch_generator(
        self,
        split: Literal["train", "val", "test"],
        desc: str = "",
        num_batches: Optional[int] = None,
    ) -> Generator[Tuple[List[torch.Tensor], torch.Tensor, int], None, None]:
        loader_dict = {"train": self.train_loader, "val": self.val_loader, "test": self.test_loader}
        loader = loader_dict[split]
        num_batches = num_batches or len(loader)
        for batch_idx, batch in tqdm.tqdm(zip(range(num_batches), loader), desc=desc):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            yield batch_idx, inputs, targets

    def _update_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor, split: Literal["train", "val", "test"]
    ) -> None:
        """The outputs input is expected to either be probabilities or single predictions."""
        for metric in self.metrics[split].values():
            metric.update(outputs, targets)

    def _compute_metrics(self, split: Literal["train", "val", "test"]) -> Dict[str, Any]:
        computed_metrics = {
            name: metric.compute().item() for name, metric in self.metrics[split].items()
        }
        return computed_metrics

    def _reset_metrics(self, split: Literal["train", "val", "test"]) -> None:
        for metric in self.metrics[split].values():
            metric.reset()

    def _report_metrics(
        self,
        split: Literal["train", "val", "test"],
        additional_metrics: Optional[Dict[str, Any]] = None,
        compute_default_metrics: bool = True,
    ) -> None:
        assert split in {"train", "val", "test"}, 'split must be one of "train", "val", "test"'
        additional_metrics = additional_metrics or {}
        computed_metrics = {} if not compute_default_metrics else self._compute_metrics(split)
        conflicted_keys = set(additional_metrics) & set(computed_metrics)
        if conflicted_keys:
            raise ValueError(
                f"additional_metrics/train_metrics conflicting keys: {conflicted_keys}"
            )
        reported_metrics = {**additional_metrics, **computed_metrics}
        # Remove any None-valued metrics with a warning (these would throw errors). Include other
        # relevant data as needed.
        for key in list(reported_metrics):
            if reported_metrics[key] is None:
                logging.warning(f"Removing metric {key} whose value is None.")
                reported_metrics.pop(key)
        if split == "train":
            report_fn = self.core_context.train.report_training_metrics
        else:
            report_fn = self.core_context.train.report_validation_metrics
        report_fn(steps_completed=self.trained_batches, metrics=reported_metrics)

    def _restore(self) -> None:
        pass

    def _save(self) -> None:
        pass

    def _train_one_batch(self, batch_idx: int, inputs: TorchData, targets: TorchData) -> None:
        outputs = self.model_engine(inputs)
        loss = self.criterion(outputs, targets)
        self.model_engine.backward(loss)
        self.model_engine.step()
        self._update_metrics(outputs=outputs, targets=targets, split="train")

    def train(self) -> None:
        if self.latest_checkpoint is not None:
            self._restore()
        for op in self.core_context.searcher.operations():
            while self.steps_completed < op.length:
                for batch_idx, inputs, targets in self._batch_generator(split="train"):
                    self._train_one_batch(batch_idx=batch_idx, inputs=inputs, targets=targets)
                self.steps_completed += 1
                if self.is_chief:
                    self._report_metrics(split="train")
                self._reset_metrics(split="train")
                self._save()
