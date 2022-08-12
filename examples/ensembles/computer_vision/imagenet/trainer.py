import logging
import json
from typing import Any, Dict, Generator, Literal, Optional, Tuple


import torch
import torch.distributed as dist
import torch.nn as nn
import torchmetrics
from attrdict import AttrDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import ensemble_metrics


class Trainer:
    def __init__(
        self,
        core_context,
        info,
        model_class: nn.Module,
        optimizer_class: torch.optim.Optimizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        hparams: AttrDict,
    ) -> None:
        self.core_context = core_context
        self.rank = core_context.distributed.rank
        self.local_rank = core_context.distributed.local_rank
        self.size = core_context.distributed.size
        self.is_distributed = self.size > 1
        self.is_chief = self.rank == 0
        self.is_local_chief = self.local_rank == 0
        self.device = "cpu" if not self.is_distributed else f"cuda:{self.rank}"

        self.model = model_class(device=self.device, **hparams.model)
        self.optimizer = optimizer_class(self.model.parameters(), **hparams.optimizer)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.worker_train_batch_size = hparams.trainer.worker_train_batch_size
        self.worker_val_batch_size = hparams.trainer.worker_val_batch_size
        self.train_metric_agg_rate = hparams.trainer.train_metric_agg_rate
        self.max_len_unit = hparams.trainer.max_len_unit

        if self.is_distributed:
            dist.init_process_group("nccl")
            self.model = DDP(self.model, device_ids=[self.rank])

        # TODO: When DDP is used, stat_dicts are saved with `module.` in front of every usual key,
        # which can lead to errors when loading non-distributed checkpoints into distributed models
        # and vice versa. Fix.
        if info.latest_checkpoint is None:
            self.trained_batches = 0
            self.completed_epochs = 0
        else:
            with core_context.checkpoint.restore_path(info.latest_checkpoint) as path:
                with open(path.joinpath("metadata.json"), "r") as f:
                    metadata_dict = json.load(f)
                self.trained_batches = metadata_dict["trained_batches"]
                self.completed_epochs = metadata_dict["completed_epochs"]
                model_state_dict = torch.load(path.joinpath("model_state_dict.pth"))
                self.model.load_state_dict(model_state_dict)
                optimizer_state_dict = torch.load(path.joinpath("optimizer_state_dict.pth"))
                self.optimizer.load_state_dict(optimizer_state_dict)

        self.train_loader = self.build_data_loader(train=True)
        self.val_loader = self.build_data_loader(train=False)
        self.criterion = nn.CrossEntropyLoss()
        train_metrics = {
            **{f"train_top{k}_acc": torchmetrics.Accuracy(top_k=k) for k in range(1, 6)},
            "train_loss": ensemble_metrics.CrossEntropyMean(),
        }
        val_metrics = {
            **{f"val_top{k}_acc": torchmetrics.Accuracy(top_k=k) for k in range(1, 6)},
            "val_loss": ensemble_metrics.CrossEntropyMean(),
        }
        for met in train_metrics.values():
            met.to(self.device)
        for met in val_metrics.values():
            met.to(self.device)
        self.metrics = {"train": train_metrics, "val": val_metrics}

    def build_data_loader(self, train: bool) -> DataLoader:
        dataset = self.train_dataset if train else self.val_dataset
        if self.is_chief:
            logging.info(
                f"Building {'train' if train else 'val'} data loader. Records: {len(dataset)}"
            )
        worker_batch_size = self.worker_train_batch_size if train else self.worker_val_batch_size
        if self.is_distributed:
            sampler = DistributedSampler(dataset, shuffle=train)
        else:
            sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
        loader = DataLoader(dataset, batch_size=worker_batch_size, sampler=sampler, drop_last=train)
        return loader

    def batch_generator(
        self, train: bool, batches_per_epoch: Optional[int] = None
    ) -> Generator[Tuple[int, torch.Tensor, torch.Tensor], None, None]:
        loader = self.train_loader if train else self.val_loader
        if batches_per_epoch is None:
            batches_per_epoch = len(loader)
        assert batches_per_epoch <= len(loader), "batches_per_epoch exceeds batches in the loader."
        for batch_idx, batch in zip(range(batches_per_epoch), loader):
            inputs, labels = batch
            labels = labels.to(self.device)
            if isinstance(inputs, list):
                inputs = [inpt.to(self.device) for inpt in inputs]
            else:
                inputs = inputs.to(self.device)
            yield batch_idx, inputs, labels

    def train_one_epoch(self) -> None:
        """Train the model for one epoch."""
        self.model.train()
        if self.is_chief:
            print(80 * "*", f"Training: epoch {self.completed_epochs}", 80 * "*", sep="\n")
        for batch_idx, inputs, labels in self.batch_generator(train=True):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.get_loss_and_update_metrics(outputs, labels, split="train")
            loss.backward()
            self.optimizer.step()
            self.trained_batches += 1
            if self.trained_batches % self.train_metric_agg_rate == 0:
                computed_metrics = self.compute_metrics("train")
                if self.is_chief:
                    self.core_context.train.report_training_metrics(
                        steps_completed=self.trained_batches, metrics=computed_metrics
                    )
                self.reset_metrics("train")

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
                self.get_loss_and_update_metrics(outputs, labels, split="val")
            val_metrics = self.compute_metrics("val")
            if self.is_chief:
                self.core_context.train.report_validation_metrics(
                    steps_completed=self.trained_batches, metrics=val_metrics
                )
            self.reset_metrics("val")
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
                op.report_completed(val_metrics["val_top1_acc"])

    def get_loss_and_update_metrics(self, outputs, labels, split=Literal["train", "val"]):
        loss = self.criterion(outputs, labels)
        for met in self.metrics[split].values():
            met(outputs, labels)
        return loss

    def compute_metrics(self, split: Literal["train", "val"]) -> Dict[str, Any]:
        split_metrics = self.metrics[split]
        computed_metrics = {name: metric.compute().item() for name, metric in split_metrics.items()}
        return computed_metrics

    def reset_metrics(self, split: Literal["train", "val"]) -> None:
        split_metrics = self.metrics[split]
        for metric in split_metrics.values():
            metric.reset()

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
