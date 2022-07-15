from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchmetrics


class EnsembleTrainer(nn.Module):
    def __init__(
        self,
        core_context,
        model_list: List[nn.Module],
        train_batch_size: int,
        val_batch_size: int,
        val_dataset: Dataset,
        train_dataset: Optional[Dataset] = None,
        ensemble_strategy: str = "naive",
        ensemble_args: Optional[dict] = None,
        extra_val_log_metrics: Dict[str, Any] = None,
        sanity_check: bool = False,
    ) -> None:
        super().__init__()
        self.core_context = core_context
        self.models = nn.ModuleList(model_list)
        self.models.eval()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.ensemble_strategy = ensemble_strategy
        self.ensemble_args = ensemble_args or {}
        self.extra_val_log_metrics = extra_val_log_metrics or {}
        self.sanity_check = sanity_check
        if self.sanity_check:
            print(f"Running in sanity check mode!")

        self.rank = core_context.distributed.rank
        self.is_distributed = core_context.distributed.size > 1
        self.is_chief = self.rank == 0
        self.device = f"cuda:{self.rank}"
        # Move, freeze, and eval, all models.
        self.models.to(self.device)
        for param in self.models.parameters():
            param.requires_grad = False
        self.models.eval()
        if self.is_distributed:
            dist.init_process_group("nccl")
            self.models = DDP(self.models, device_ids=[self.rank])

        self.trained_batches = 0
        self.train_loader = self.build_train_loader()
        self.val_loader = self.build_val_loader()
        self.criterion = nn.NLLLoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=0.001)

        self._ensemble_strategies = {"naive": self._build_naive_ensemble}
        self.ensemble_weights = None

        self.accuracy_metrics = {
            f"top{k}_acc": torchmetrics.Accuracy(top_k=k) for k in range(1, 11)
        }
        for met in self.accuracy_metrics.values():
            met.to(self.device)
        self.loss_metric = torchmetrics.MeanMetric()
        self.loss_metric.to(self.device)

    def build_train_loader(self) -> DataLoader:
        # Not every ensembling strategy needs a train loader
        if self.train_dataset is None:
            return
        if self.is_distributed:
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
        else:
            sampler = RandomSampler(self.train_dataset)
        loader = DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, sampler=sampler, drop_last=True
        )
        return loader

    def build_val_loader(self) -> DataLoader:
        if self.is_distributed:
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            sampler = SequentialSampler(self.val_dataset)
        loader = DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, sampler=sampler, drop_last=True
        )
        return loader

    def build_ensemble(self) -> None:
        self._ensemble_strategies[self.ensemble_strategy](*self.ensemble_args)

    def _build_naive_ensemble(self) -> None:
        num_models = len(self.models)
        self.ensemble_weights = torch.ones(num_models, device=self.device) / num_models

    def forward(self, inputs):
        """Returns probabilties for the ensembled model."""
        model_probs = torch.stack([model(inputs).softmax(dim=1) for model in self.models], dim=-1)
        ensemble_prob = model_probs @ self.ensemble_weights
        if self.sanity_check:
            prob_sum_check = ensemble_prob.sum(dim=1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        return ensemble_prob

    def validate_ensemble(self) -> None:
        self.models.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                probs = self(inputs)
                self.get_loss_and_update_metrics(probs, labels)
            if self.is_chief:
                computed_metrics = self.compute_metrics("val_")
                conflicted_keys = set(self.extra_val_log_metrics) & set(computed_metrics)
                if conflicted_keys:
                    raise ValueError(
                        f"extra_val_log_metrics/val_metrics conflicting keys: {conflicted_keys}"
                    )
                reported_metrics = {**self.extra_val_log_metrics, **computed_metrics}
                self.core_context.train.report_validation_metrics(
                    steps_completed=self.trained_batches, metrics=reported_metrics
                )
            self.reset_metrics()
        if self.core_context.preempt.should_preempt():
            return

    def get_loss_and_update_metrics(self, probs, labels):
        for met in self.accuracy_metrics.values():
            met(probs, labels)
        # NLLLoss expects log-probabilities
        loss = self.criterion(probs.log(), labels)
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
