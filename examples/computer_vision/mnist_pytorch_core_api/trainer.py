import json
from typing import Any, Dict, Generator, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import Accuracy, MeanMetric


class Trainer:
    def __init__(
        self,
        core_context,
        info,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        worker_train_batch_size: int,
        worker_val_batch_size: int,
        train_metric_agg_rate: int,
    ) -> None:
        self.core_context = core_context
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.worker_train_batch_size = worker_train_batch_size
        self.worker_val_batch_size = worker_val_batch_size
        self.train_metric_agg_rate = train_metric_agg_rate

        self.rank = core_context.distributed.rank
        self.local_rank = core_context.distributed.local_rank
        self.size = core_context.distributed.size
        self.is_distributed = self.size > 1
        self.is_chief = self.rank == 0
        self.is_local_chief = self.local_rank == 0
        self.device = "cpu" if not self.is_distributed else f"cuda:{self.rank}"
        self.model.to(self.device)

        if info.latest_checkpoint is None:
            self.trained_batches = 0
            self.completed_epochs = 0
        else:
            with core_context.checkpoint.restore_path(info.latest_checkpoint) as path:
                with open(path.joinpath("metadata.json"), "r") as f:
                    metadata_dict = json.load(f)
                print("CHECKPOINT METADATA:", metadata_dict)
                self.trained_batches = metadata_dict["trained_batches"]
                self.completed_epochs = metadata_dict["completed_epochs"]
                model_state_dict = torch.load(path.joinpath("model_state_dict.pth"))
                self.model.load_state_dict(model_state_dict)
                optimizer_state_dict = torch.load(path.joinpath("optimizer_state_dict.pth"))
                self.optimizer.load_state_dict(optimizer_state_dict)

        if self.is_distributed:
            dist.init_process_group("nccl")
            self.model = DDP(self.model, device_ids=[self.rank])

        self.train_loader = self.build_data_loader(train=True)
        self.val_loader = self.build_data_loader(train=False)
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy_metrics = {f"top{k}_acc": Accuracy(top_k=k) for k in range(1, 6)}
        for met in self.accuracy_metrics.values():
            met.to(self.device)
        self.loss_metric = MeanMetric()
        self.loss_metric.to(self.device)

    def build_data_loader(self, train: bool) -> DataLoader:
        dataset = self.train_dataset if train else self.val_dataset
        worker_batch_size = self.worker_train_batch_size if train else self.worker_val_batch_size
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
        for batch_idx, batch in enumerate(loader):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            yield batch_idx, inputs, labels

    def train_one_epoch(self) -> None:
        """Train the model for one epoch."""
        self.model.train()
        if self.is_chief:
            print(80 * "*", f"Training: epoch {self.completed_epochs}", 80 * "*", sep="\n")
        for batch_idx, inputs, labels in self.batch_generator(train=True):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.get_loss_and_update_metrics(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.trained_batches += 1
            if self.trained_batches % self.train_metric_agg_rate == 0:
                computed_metrics = self.compute_metrics("train_")
                if self.is_chief:
                    self.core_context.train.report_training_metrics(
                        steps_completed=self.trained_batches, metrics=computed_metrics
                    )
                self.reset_metrics()

    def validate(self) -> Dict[str, Any]:
        "Evaluate the model on the validation set and return all validation metrics."
        self.model.eval()
        if self.is_chief:
            print(
                80 * "*",
                f"Validating: epoch {self.completed_epochs}",
                80 * "*",
                sep="\n",
            )
        with torch.no_grad():
            for batch_idx, inputs, labels in self.batch_generator(train=False):
                outputs = self.model(inputs)
                self.get_loss_and_update_metrics(outputs, labels)
            val_metrics = self.compute_metrics("val_")
            if self.is_chief:
                self.core_context.train.report_validation_metrics(
                    steps_completed=self.trained_batches, metrics=val_metrics
                )
            self.reset_metrics()
        return val_metrics

    def train(self) -> None:
        for op in self.core_context.searcher.operations():
            while self.completed_epochs < op.length:
                self.train_one_epoch()
                val_metrics = self.validate()
                # Update completed_epochs before checkpointing for accurate checkpoint metadata
                self.completed_epochs += 1
                self.checkpoint()
                if self.core_context.preempt.should_preempt():
                    return
            if self.is_chief:
                op.report_completed(val_metrics["val_loss"])

    def get_loss_and_update_metrics(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        for met in self.accuracy_metrics.values():
            met(outputs, labels)
        self.loss_metric(loss)
        return loss

    def compute_metrics(self, prefix: str = ""):
        computed_metrics = {
            prefix + name: metric.compute().item() for name, metric in self.accuracy_metrics.items()
        }
        computed_metrics[prefix + "loss"] = self.loss_metric.compute().item()
        return computed_metrics

    def reset_metrics(self):
        for met in self.accuracy_metrics.values():
            met.reset()
        self.loss_metric.reset()

    def checkpoint(self) -> None:
        if self.is_chief:
            checkpoint_metadata = {
                "completed_epochs": self.completed_epochs,
                "trained_batches": self.trained_batches,
                "steps_completed": self.trained_batches,
            }
            with self.core_context.checkpoint.store_path(checkpoint_metadata) as (path, storage_id):
                torch.save(self.model.state_dict(), path.joinpath("model_state_dict.pth"))
                torch.save(self.optimizer.state_dict(), path.joinpath("optimizer_state_dict.pth"))
