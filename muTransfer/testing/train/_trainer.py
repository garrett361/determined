from typing import Any, Callable, Dict, Generator, Tuple

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
    ) -> None:
        self.core_context = core_context
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.metric_agg_rate = metric_agg_rate

        self.trained_batches = 0
        self.rank = core_context.distributed.rank
        self.local_rank = core_context.distributed.local_rank
        self.size = core_context.distributed.size
        self.is_distributed = self.size > 1
        self.is_chief = self.rank == 0
        self.is_local_chief = self.local_rank == 0
        self.device = "cpu" if not self.is_distributed else f"cuda:{self.rank}"
        self.model.to(self.device)

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
                if self.trained_batches % self.metric_agg_rate == 0:
                    computed_metrics = self.compute_metrics()
                    if self.is_chief:
                        self.core_context.train.report_training_metrics(
                            steps_completed=self.trained_batches, metrics=computed_metrics
                        )
                    self.reset_metrics()
                if self.core_context.preempt.should_preempt():
                    return
                if self.trained_batches == op.length:
                    if self.is_chief:
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
