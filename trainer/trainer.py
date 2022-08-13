import json
import logging
from typing import Any, Dict, Generator, Tuple

import attrdict
import determined as det
import torch
import torch.distributed as dist
import torch.nn as nn
import torchmetrics
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)


class Trainer:
    def __init__(
        self,
        model_class: nn.Module,
        optimizer_class: torch.optim.Optimizer,
        criterion_class: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> None:

        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.criterion_class = criterion_class
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.info = det.get_cluster_info()

        # Updated if a checkpoint is restored:
        # TODO: Currently, steps_completed is the number of trained batches. Add more flexibility.
        self.steps_completed = 0
        self.completed_epochs = 0

        # Instantiated later in _setup():
        self.hparams = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.rank = None
        self.local_rank = None
        self.size = None
        self.is_distributed = None
        self.is_chief = None
        self.is_local_chief = None
        self.device = None
        self.metrics = {"train": None, "val": None}

    def build_metrics(self, train: bool) -> Dict[str, Any]:
        """Build dictionary of training metrics."""
        # For simplicity, using the same metrics for train and val and only using accuracy metrics.
        metric_prefix = "train" if train else "val"
        metrics = {
            f"{metric_prefix}_top{k}_acc": torchmetrics.Accuracy(top_k=k) for k in range(1, 6)
        }
        return metrics

    def build_data_loader(self, train: bool) -> DataLoader:
        dataset = self.train_dataset if train else self.val_dataset
        worker_batch_size = (
            self.hparams.trainer.worker_train_batch_size
            if train
            else self.hparams.trainer.worker_val_batch_size
        )
        if self.is_distributed:
            sampler = DistributedSampler(dataset, shuffle=train)
        else:
            sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
        loader = DataLoader(dataset, batch_size=worker_batch_size, sampler=sampler, drop_last=train)
        return loader

    def batch_generator(
        self, train: bool
    ) -> Generator[Tuple[int, torch.Tensor, torch.Tensor], None, None]:
        loader = self.train_loader if train else self.val_loader
        for batch in loader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            yield inputs, targets

    def train_one_epoch(self, core_context: det.core.Context) -> None:
        """Train the model for one epoch."""
        self.model.train()
        if self.is_chief:
            print(80 * "*", f"Training: epoch {self.completed_epochs}", 80 * "*", sep="\n")
        for inputs, targets in self.batch_generator(train=True):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # TODO: Support gradient aggregation.
            loss = self.get_loss_and_update_metrics(outputs, targets, train=True)
            loss.backward()
            self.optimizer.step()
            if self.steps_completed % self.hparams.trainer.train_metric_agg_rate == 0:
                self.get_metrics(train=True, core_context=core_context)

    def validate(self, core_context: det.core.Context) -> Dict[str, Any]:
        """Evaluate the model on the validation set and return all validation metrics."""
        if self.is_chief:
            print(80 * "*", f"Validation epoch {self.completed_epochs}", 80 * "*", sep="\n")
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.batch_generator(train=False):
                outputs = self.model(inputs)
                self.get_loss_and_update_metrics(outputs, targets, train=False)
        val_metrics = self.get_metrics(train=False, core_context=core_context)
        self.completed_epochs += 1
        return val_metrics

    def run(self) -> None:
        with self._get_core_context() as core_context:
            self._setup(core_context=core_context)
            for op in core_context.searcher.operations():
                # TODO: Support non-epoch based loops.
                while self.completed_epochs < op.length:
                    self.train_one_epoch(core_context=core_context)
                    val_metrics = self.validate(core_context=core_context)
                    self.save_checkpoint(core_context=core_context)
                    if core_context.preempt.should_preempt():
                        return
                if self.is_chief:
                    # TODO: Don't hard code the searcher metric; get from searcher config.
                    op.report_completed(val_metrics["val_top1_acc"])

    def get_loss_and_update_metrics(self, outputs, targets, train: bool):
        metrics = self.metrics["train" if train else "val"]
        for metric in metrics.values():
            metric.update(outputs, targets)
        loss = self.criterion(outputs, targets)
        if train:
            # This call will need to be moved if gradient aggregation is supported.
            self.steps_completed += 1
        return loss

    def _setup(self, core_context: det.core.Context) -> None:
        self.rank = core_context.distributed.rank
        self.local_rank = core_context.distributed.local_rank
        self.size = core_context.distributed.size
        self.is_distributed = self.size > 1
        self.is_chief = self.rank == 0
        self.is_local_chief = self.local_rank == 0
        self.device = "cpu" if not self.is_distributed else f"cuda:{self.rank}"

        self.metrics = {
            "train": self.build_metrics(train=True),
            "val": self.build_metrics(train=False),
        }
        for split in self.metrics.values():
            for metric in split.values():
                metric.to(self.device)

        self.hparams = attrdict.AttrDict(self.info.trial.hparams)
        self.model = self.model_class(**self.hparams.model)
        self.model.to(self.device)
        self.optimizer = self.optimizer_class(self.model.parameters(), **self.hparams.optimizer)
        if self.is_distributed:
            dist.init_process_group("nccl")
            # Important to call DDP before _restore_latest_checkpoint (below) due to DDP's effects
            # on torch's state_dict naming conventions.
            self.model = DDP(self.model, device_ids=[self.rank])

        self.optimizer = self.optimizer_class(self.model.parameters(), **self.hparams.optimizer)
        self.criterion = self.criterion_class(**self.hparams.criterion)

        self._restore_latest_checkpoint(core_context=core_context)

        self.train_loader = self.build_data_loader(train=True)
        self.val_loader = self.build_data_loader(train=False)

    def _restore_latest_checkpoint(self, core_context: det.core.Context) -> None:
        """Restores the experiment state to the latest saved checkpoint, if it exists."""
        if self.info.latest_checkpoint is not None:
            with core_context.checkpoint.restore_path(self.info.latest_checkpoint) as path:
                with open(path.joinpath("metadata.json"), "r") as f:
                    metadata_dict = json.load(f)
                self.steps_completed = metadata_dict["steps_completed"]
                self.completed_epochs = metadata_dict["completed_epochs"]
                model_state_dict = torch.load(path.joinpath("model_state_dict.pth"))
                self.model.load_state_dict(model_state_dict)
                optimizer_state_dict = torch.load(path.joinpath("optimizer_state_dict.pth"))
                self.optimizer.load_state_dict(optimizer_state_dict)

    def get_metrics(self, train: bool, core_context=det.core.Context):
        """Computes, reports, and resets all relevant metrics."""
        metrics = self.metrics["train" if train else "val"]
        computed_metrics = {name: metric.compute().item() for name, metric in metrics.items()}
        report_fn = (
            core_context.train.report_training_metrics
            if train
            else core_context.train.report_validation_metrics
        )
        if self.is_chief:
            # TODO: Support non-batch-based notions of steps_completed
            report_fn(steps_completed=self.steps_completed, metrics=computed_metrics)
        for met in metrics.values():
            met.reset()
        return computed_metrics

    def save_checkpoint(self, core_context: det.core.Context) -> None:
        if self.is_chief:
            checkpoint_metadata = {
                "completed_epochs": self.completed_epochs,
                "steps_completed": self.steps_completed,
            }
            with core_context.checkpoint.store_path(checkpoint_metadata) as (path, storage_id):
                torch.save(self.model.state_dict(), path.joinpath("model_state_dict.pth"))
                torch.save(self.optimizer.state_dict(), path.joinpath("optimizer_state_dict.pth"))

    def _get_core_context(self) -> det.core.Context:
        try:
            distributed = det.core.DistributedContext.from_torch_distributed()
        except KeyError:
            distributed = None
        core_context = det.core.init(distributed=distributed)
        return core_context
