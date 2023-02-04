import json
import logging
from typing import Any, Callable, Dict, Generator, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import MeanMetric


class Trainer:
    """A super minimal trainer class. Trains for a fixed number of batches. No checkpointing. Only
    computes and reports loss."""

    def __init__(
        self,
        core_context,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        criterion: Callable,
        batch_size: int = 2 ** 7,
        metric_agg_rate: int = 10,
        latest_checkpoint: Optional[str] = None,
    ) -> None:
        self.core_context = core_context
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.metric_agg_rate = metric_agg_rate

        self.rank = core_context.distributed.rank
        self.local_rank = core_context.distributed.local_rank
        self.size = core_context.distributed.size
        self.is_distributed = self.size > 1
        self.is_chief = self.rank == 0
        self.is_local_chief = self.local_rank == 0
        self.device = "cpu" if not self.is_distributed else f"cuda:{self.rank}"
        self.model.to(self.device)

        if latest_checkpoint is None:
            self.trained_batches = 0
        else:
            with core_context.checkpoint.restore_path(latest_checkpoint) as path:
                with open(path.joinpath("metadata.json"), "r") as f:
                    metadata_dict = json.load(f)
                self.trained_batches = metadata_dict["trained_batches"]
                model_state_dict = torch.load(
                    path.joinpath("model_state_dict.pth"), map_location=self.device
                )
                self.model.load_state_dict(
                    model_state_dict,
                )
                optimizer_state_dict = torch.load(
                    path.joinpath("optimizer_state_dict.pth"), map_location=self.device
                )
                self.optimizer.load_state_dict(optimizer_state_dict)

        if self.is_distributed:
            dist.init_process_group("nccl")
            self.model = DDP(self.model, device_ids=[self.rank])

        self.dataloader = self.build_dataloader()
        self.criterion = criterion

        self.loss_metric = MeanMetric()
        self.loss_metric.to(self.device)

    def build_dataloader(self) -> DataLoader:
        if self.is_distributed:
            sampler = DistributedSampler(self.dataset)
        else:
            sampler = RandomSampler(self.dataset)
        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=sampler, drop_last=True
        )
        return loader

    def batch_generator(self) -> Generator[Tuple[int, torch.Tensor, torch.Tensor], None, None]:
        while True:
            for batch in self.dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                yield inputs, labels

    def train(self) -> None:
        self.model.train()
        for op in self.core_context.searcher.operations():
            for inputs, labels in self.batch_generator():
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.get_loss_and_update_metrics(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.trained_batches += 1
                finishing_training = self.trained_batches == op.length
                should_compute_metrics_and_checkpoint = (
                    self.trained_batches % self.metric_agg_rate == 0 or finishing_training
                )
                if should_compute_metrics_and_checkpoint:
                    computed_metrics = self.compute_metrics()
                    if self.is_chief:
                        self.core_context.train.report_training_metrics(
                            steps_completed=self.trained_batches, metrics=computed_metrics
                        )
                    self.reset_metrics()
                    self.checkpoint()
                if self.core_context.preempt.should_preempt():
                    return
                if finishing_training:
                    if self.is_chief:
                        self.core_context.train.report_validation_metrics(
                            steps_completed=self.trained_batches, metrics=computed_metrics
                        )
                        op.report_completed(computed_metrics["loss"])
                    return

    def get_loss_and_update_metrics(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        self.loss_metric(loss)
        return loss

    def compute_metrics(self, prefix: str = "") -> Dict[str, Any]:
        computed_metrics = {prefix + "loss": self.loss_metric.compute().item()}
        return computed_metrics

    def reset_metrics(self):
        self.loss_metric.reset()

    def checkpoint(self):
        if self.is_chief:
            checkpoint_metadata = {
                "trained_batches": self.trained_batches,
                "steps_completed": self.trained_batches,
            }
            with self.core_context.checkpoint.store_path(checkpoint_metadata) as (
                path,
                _,
            ):
                torch.save(self.model.state_dict(), path.joinpath("model_state_dict.pth"))
                torch.save(
                    self.optimizer.state_dict(),
                    path.joinpath("optimizer_state_dict.pth"),
                )
