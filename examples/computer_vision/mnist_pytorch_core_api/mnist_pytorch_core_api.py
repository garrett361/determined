import logging

import determined as det
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as T
import uuid
from attrdict import AttrDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import Accuracy, MeanMetric


class MNISTModel(nn.Module):
    def __init__(
        self,
        n_filters1: int = 32,
        n_filters2: int = 64,
        dropout1: float = 0.25,
        dropout2: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, n_filters1, 3, 1),
            nn.ReLU(),
            nn.Conv2d(
                n_filters1,
                n_filters2,
                3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout1),
            nn.Flatten(),
            nn.Linear(144 * n_filters2, 128),
            nn.ReLU(),
            nn.Dropout2d(dropout2),
            nn.Linear(128, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class Trainer:
    def __init__(
        self,
        core_context,
        info,
        model: nn.Module,
        worker_train_batch_size: int,
        worker_val_batch_size: int,
        metric_agg_rate_batches: int,
        epochs: int,
        val_freq: int,
    ) -> None:
        self.core_context = core_context
        self.model = model
        self.worker_train_batch_size = worker_train_batch_size
        self.worker_val_batch_size = worker_val_batch_size
        self.metric_agg_rate_batches = metric_agg_rate_batches
        self.metric_agg_rate_batches = metric_agg_rate_batches
        self.epochs = epochs
        self.val_freq = val_freq

        if info.latest_checkpoint is None:
            self.trained_batches = 0
            self.trained_epochs = 0
        else:
            with core_context.checkpoint.restore_path(info.latest_checkpoint) as path:
                print("CHECKPOINT PATH", path)
                self.trained_batches = 0
                self.trained_epochs = 0

        self.rank = core_context.distributed.rank
        self.local_rank = core_context.distributed.local_rank
        self.size = core_context.distributed.size
        self.is_distributed = self.size > 1
        self.is_chief = self.rank == 0
        self.is_local_chief = self.local_rank == 0
        self.device = f"cuda:{self.rank}"
        self.model.to(self.device)
        if self.is_distributed:
            dist.init_process_group("nccl")
            self.model = DDP(self.model, device_ids=[self.rank])

        self.trained_batches = 0
        self.train_loader = self.build_data_loader(train=True)
        self.val_loader = self.build_data_loader(train=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        self.accuracy_metrics = {f"top{k}_acc": Accuracy(top_k=k) for k in range(1, 6)}
        for met in self.accuracy_metrics.values():
            met.to(self.device)
        self.loss_metric = MeanMetric()
        self.loss_metric.to(self.device)

    def build_data_loader(self, train: bool) -> DataLoader:
        mnist_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ]
        )
        worker_batch_size = self.worker_train_batch_size if train else self.worker_val_batch_size
        dataset = datasets.MNIST(
            root="shared_fs/data", train=train, download=False, transform=mnist_transform
        )
        if self.is_distributed:
            sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
        loader = DataLoader(dataset, batch_size=worker_batch_size, sampler=sampler, drop_last=True)
        return loader

    def train_one_epoch(self) -> None:
        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            self.trained_batches += 1
            images, labels = batch
            print(f"{self.rank} BATCH_SIZE {images.shape[0]}")
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.get_loss_and_update_metrics(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if (self.trained_batches + 1) % self.metric_agg_rate_batches == 0:
                computed_metrics = self.compute_metrics("train_")
                if self.is_chief:
                    core_context.train.report_training_metrics(
                        steps_completed=self.trained_batches, metrics=computed_metrics
                    )
                self.reset_metrics()
        if self.core_context.preempt.should_preempt():
            return

    def validate(self) -> None:
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                images, labels = batch
                print(f"{self.rank} BATCH_SIZE {images.shape[0]}")
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                self.get_loss_and_update_metrics(outputs, labels)
                computed_metrics = self.compute_metrics("val_")
                computed_metrics["test_list"] = list(range(10))
            if self.is_chief:
                print(80 * "=", "reporting val metrics", 80 * "=", sep="\n")
                core_context.train.report_validation_metrics(
                    steps_completed=self.trained_batches, metrics=computed_metrics
                )
            self.reset_metrics()
        self.checkpoint()
        if self.core_context.preempt.should_preempt():
            return

    def train(self) -> None:
        for epoch_idx in range(self.epochs):
            if self.is_chief:
                print(80 * "*", f"Training during epoch {epoch_idx}", 80 * "*", sep="\n")
            self.train_one_epoch()
            if (epoch_idx + 1) % self.val_freq == 0:
                if self.is_chief:
                    print(80 * "*", f"Validating during epoch {epoch_idx}", 80 * "*", sep="\n")
                self.validate()

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
        storage_manager = self.core_context.checkpoint._storage_manager
        storage_id = str(uuid.uuid4())
        if self.is_chief:
            with storage_manager.store_path(storage_id) as path:
                print(f"Saving model to {path}")
                # Broadcast checkpoint path to all ranks.
                self.core_context.distributed.broadcast((storage_id, path))
                torch.save({"test": 0}, path)
                # Gather resources across nodes.
                all_resources = self.core_context.distributed.gather(
                    storage_manager._list_directory(path)
                )
            resources = {k: v for d in all_resources for k, v in d.items()}

            self.core_context.checkpoint._report_checkpoint(storage_id, resources)
        else:
            storage_id, path = self.core_context.distributed.broadcast(None)
            torch.save(self.model.state_dict(), path)
            # Gather resources across nodes.
            _ = self.core_context.distributed.gather(storage_manager._list_directory(path))
            if self.is_local_chief:
                storage_manager.post_store_path(storage_id, path)


def main(core_context, info, hparams: AttrDict, model_class: nn.Module = MNISTModel) -> None:
    model = model_class(**hparams.model)
    trainer = Trainer(core_context, info, model, **hparams.trainer)
    trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    hparams = AttrDict(info.trial.hparams)

    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context, info, hparams)
