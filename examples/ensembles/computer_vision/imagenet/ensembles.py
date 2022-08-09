import logging
import random
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchmetrics
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import data
import ensemble_metrics
import strategies


class ClassificationEnsemble(nn.Module):
    """
    Class for quickly training and validating various ensemble strategies.
    TODO: Currently assuming each model gets a differently transformed input; can optimize when
    multiple transforms are the same.
    """

    def __init__(
        self,
        core_context,
        models: List[nn.Module],
        transforms: Union[Callable, List[Callable]],
        train_batch_size: int,
        val_batch_size: int,
        dataset_name: str,
        ensemble_strategy: str,
        sanity_check: bool = False,
        num_combinations: Optional[int] = None,
        lr: Optional[float] = None,
        epochs: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.core_context = core_context
        self.models = models
        self.transforms = transforms
        self.num_models = len(self.models)
        self.models.eval()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataset_name = dataset_name
        self.ensemble_strategy = ensemble_strategy
        self.sanity_check = sanity_check
        if self.sanity_check:
            logging.info(f"Running in sanity check mode!")
        self.num_combinations = num_combinations
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed

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

        self._strategy = strategies.STRATEGY_DICT[self.ensemble_strategy](self)

        if self._strategy.requires_training:
            logging.info(f"Building train_dataset")
            self.train_dataset = data.get_dataset(
                name=self.dataset_name, split="train", transforms=self.transforms
            )
            logging.info(f"{len(self.train_dataset)} records in train_dataset")
        else:
            logging.info(f"Skipping building train_dataset")
            self.train_dataset = None
        logging.info(f"Building val_dataset")
        self.val_dataset = data.get_dataset(
            name=self.dataset_name, split="val", transforms=self.transforms
        )
        logging.info(f"{len(self.val_dataset)} records in val_dataset")
        self.train_loader = self.build_train_loader()
        self.val_loader = self.build_val_loader()
        self.test_loader = None
        self.trained_batches = 0
        self._build_metrics()

        # The following attributes will be set by the ensemble's Strategy instance.

        # There can be multiple notions of weights for different ensemble strategies.  `weights`
        # are generally used to weight the final individual model probabilities or
        # logits, while _other_weights cover other forms of weights.
        self.ensemble_weights = None
        self._other_weights = None
        # Others need log-likelihoods at intermediate steps.
        self._log_likelihoods = None
        # Sometimes we calibrate using an inverse temperature.
        self.betas = None
        # Sometimes we need SGD
        self.optimizer = None

    def _set_random_seeds(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.random.manual_seed(self.random_seed)

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
        loader = DataLoader(self.val_dataset, batch_size=self.val_batch_size, sampler=sampler)
        return loader

    def batch_generator(
        self,
        split: Literal["train", "val", "test"],
        desc: str = "",
        num_batches: Optional[int] = None,
    ) -> Generator[Tuple[List[torch.Tensor], torch.Tensor, int], None, None]:
        loader_dict = {"train": self.train_loader, "val": self.val_loader, "test": self.test_loader}
        loader = loader_dict[split]
        num_batches = num_batches or len(loader)
        for batch_idx, batch in tqdm.tqdm(zip(range(num_batches), loader), desc=desc):
            inputs, labels = batch
            inputs = [inpt.to(self.device) for inpt in inputs]
            labels = labels.to(self.device)
            yield inputs, labels, batch_idx

    def _build_metrics(self) -> None:
        # Create metrics for the various splits.  These all take in preds, labels pairs as args
        # where preds are either probabilities or single predictions.
        self.metrics = {"train": {}, "val": {}, "test": {}}
        # Metrics for all
        for split, met_dic in self.metrics.items():
            if self._strategy.generates_probabilities:
                for k in range(1, 11):
                    met_dic[f"{split}_top{k}_acc"] = torchmetrics.Accuracy(top_k=k)
                met_dic[f"{split}_loss"] = ensemble_metrics.NLLMeanMetric()
                met_dic[f"{split}_cal_error_l1"] = torchmetrics.CalibrationError(norm="l1")
                met_dic[f"{split}_cal_error_l2"] = torchmetrics.CalibrationError(norm="l2")
                met_dic[f"{split}_cal_error_max"] = torchmetrics.CalibrationError(norm="max")
            else:
                met_dic[f"{split}_top1_acc"] = torchmetrics.Accuracy()

        for vals in self.metrics.values():
            for metric in vals.values():
                if metric is not None:
                    metric.to(self.device)

    def update_metrics(
        self, preds: torch.Tensor, labels: torch.Tensor, split: Literal["train", "val", "test"]
    ) -> None:
        """The preds input is expected to either be probabilities or single predictions."""
        for metric in self.metrics[split].values():
            metric.update(preds, labels)

    def compute_metrics(self, split: Literal["train", "val", "test"]) -> Dict[str, Any]:
        computed_metrics = {
            name: metric.compute().item() for name, metric in self.metrics[split].items()
        }
        return computed_metrics

    def reset_metrics(self, split: Literal["train", "val", "test"]) -> None:
        for metric in self.metrics[split].values():
            metric.reset()

    def report_metrics(
        self,
        split: Literal["train", "val", "test"],
        additional_metrics: Optional[Dict[str, Any]] = None,
        compute_default_metrics: bool = True,
    ) -> None:
        assert split in {"train", "val", "test"}, 'split must be one of "train", "val", "test"'
        additional_metrics = additional_metrics or {}
        computed_metrics = {} if not compute_default_metrics else self.compute_metrics(split)
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
        elif split in {"val", "test"}:
            report_fn = self.core_context.train.report_validation_metrics
        report_fn(steps_completed=self.trained_batches, metrics=reported_metrics)

    def build_ensemble(self) -> None:
        logging.info("Building ensemble...")
        self._strategy.build()

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Returns logits for the models, stacked along the last dimension."""
        model_logits = torch.stack(
            [model(inpt) for model, inpt in zip(self.models, inputs)], dim=-1
        )
        return model_logits

    def get_ensembled_preds_from_inputs(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Returns predictions for the ensembled model."""
        logits = self(inputs)
        ensembled_preds = self._strategy.pred(logits)
        if self._strategy.generates_probabilities and self.sanity_check:
            prob_sum_check = ensembled_preds.sum(dim=1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        return ensembled_preds

    def validate_ensemble(self) -> None:
        self.models.eval()
        if self.transformer is not None:
            self.transformer.eval()
        with torch.no_grad():
            for inputs, labels, batch_idx in self.batch_generator(split="val", desc="Validating"):
                labels = labels.to(self.device)
                probs = self.get_ensembled_preds_from_inputs(inputs)
                self.update_metrics(probs, labels, split="val")
            if self.is_chief:
                self.report_metrics(split="val")
            self.reset_metrics(split="val")
        if self.core_context.preempt.should_preempt():
            return

    def train_vbmc(self) -> None:
        """Computes the posterior and resulting effective weights for the num_combination linear
        combinations of models."""
        with torch.no_grad():
            self._log_likelihoods = torch.zeros(self.num_combinations, device=self.device)
            vbmc_criterion = nn.NLLLoss(reduction="none")
            for inputs, labels, batch_idx in self.batch_generator(
                split="train", desc="Training VBMC"
            ):
                logits = self(inputs)
                probs = logits.softmax(dim=1)
                ensemble_probs = probs @ self._other_weights
                log_ensemble_probs = ensemble_probs.log()
                labels = labels[..., None]
                labels = labels.expand(-1, self.num_combinations)
                loss = vbmc_criterion(log_ensemble_probs, labels).sum(dim=0)
                # Subtract loss to get *positive* log-likelihood sum.
                self._log_likelihoods -= loss
                self.trained_batches += 1

        # Transform back to probs with the Log-Sum-Exp trick for stability.
        self._log_likelihoods -= self._log_likelihoods.max()
        posterior = self._log_likelihoods.softmax(dim=0)
        self.ensemble_weights = self._other_weights @ posterior
        if self.sanity_check:
            prob_sum_check = self.ensemble_weights.sum(dim=-1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        metrics = {
            "posterior": [p.item() for p in posterior],
            "ensemble_weights": [w.item() for w in self.ensemble_weights],
        }
        self.report_metrics(
            split="train", additional_metrics=metrics, compute_default_metrics=False
        )
        if self.core_context.preempt.should_preempt():
            return

    def calibrate_temperature(
        self,
        stop_threshold: float = 1e-4,
        max_steps_per_batch: int = 5,
        clip_magnitude: float = 0.5,
        ema_weight: float = 0.1,
    ) -> None:
        """Calibrates temperatures for all base models in the ensemble in parallel using Newton's
        method.  An exponential moving average is used for updating temperatures across batches.
        """
        self.betas = torch.ones(self.num_models, device=self.device)
        with torch.no_grad():
            beta_history = [self.betas.clone()]
            for epoch_idx in range(self.epochs):
                for inputs, labels, batch_idx in self.batch_generator(
                    split="train", desc=f"Calibrating Temperature (epoch {epoch_idx})"
                ):
                    self.trained_batches += 1
                    initial_beta = self.betas.clone()
                    for step_idx in range(max_steps_per_batch):
                        score = self(inputs)
                        probs = (self.betas * score).softmax(dim=1)
                        mean_true_score = score[torch.arange(len(labels)), labels].mean(dim=0)
                        score_label_mean = (probs * score).sum(dim=1)
                        score2_label_mean = (probs * score ** 2).sum(dim=1)
                        gradient = score_label_mean.mean(dim=0) - mean_true_score
                        hessian = (score2_label_mean - score_label_mean ** 2).mean(dim=0)
                        delta_beta = -1 * gradient / hessian
                        if delta_beta.isnan().any() or delta_beta.abs().max() < stop_threshold:
                            break
                        # Clamp to help prevent runaways due to noise
                        delta_beta = delta_beta.clamp(min=-clip_magnitude, max=clip_magnitude)
                        self.betas += delta_beta
                    # Update temperature with an exponential moving average.
                    self.betas = (1 - ema_weight) * initial_beta + ema_weight * self.betas
                    beta_history.append(self.betas.clone())
                    beta_dict = {f"beta_{idx}": b.item() for idx, b in enumerate(self.betas)}
                    if self.is_chief:
                        self.report_metrics(
                            split="train",
                            additional_metrics=beta_dict,
                            compute_default_metrics=False,
                        )
            if self.core_context.preempt.should_preempt():
                return

    def _conjugate_gradient(
        self,
        gradient: torch.Tensor,
        hessian: torch.Tensor,
        initial_guess: Optional[torch.Tensor] = None,
        stop_threshold: float = 1e-4,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.sanity_check:
                assert len(gradient.shape) == 1
                for dim_size in hessian.shape:
                    assert gradient.shape[0] == dim_size
            num_steps = gradient.shape[0]
            x_prev = initial_guess if initial_guess is not None else torch.zeros_like(gradient)
            g_prev = gradient + hessian @ x_prev
            p_prev = g_prev
            g_prev_squared = g_prev @ g_prev

            for step in range(num_steps):
                Hp_prev = hessian @ p_prev
                alpha_prev = -g_prev_squared / (p_prev @ Hp_prev)
                x_next = x_prev + alpha_prev * p_prev
                g_next = g_prev + alpha_prev * Hp_prev
                g_next_squared = g_next @ g_next
                if step == num_steps - 1:
                    if self.sanity_check:
                        residual = gradient + hessian @ x_next
                        residual_norm = residual.norm()
                        if residual_norm > stop_threshold:
                            logging.warning(
                                f"Conjugate gradient residual larger "
                                f"than {stop_threshold}: {residual_norm.item()}:"
                            )
                    return x_next
                if g_next_squared.sqrt() < stop_threshold:
                    return x_next
                beta_prev = g_next_squared / g_prev_squared
                p_prev = g_next + beta_prev * p_prev
                g_prev = g_next
                g_prev_squared = g_next_squared
                x_prev = x_next

    def train_ensemble_weights_with_conjugate_gradient(
        self,
        initial_guess: Optional[torch.Tensor] = None,
        stop_threshold: float = 1e-4,
        max_steps_per_batch: int = 5,
        clip_magnitude: float = 1.0,
        ema_weight: float = 0.1,
    ) -> None:
        """Trains ensemble weights using conjugate gradient. An exponential moving average is used
        for updating temperatures across batches.
        """
        with torch.no_grad():
            ensemble_weight_history = [self.ensemble_weights.clone()]
            for epoch_idx in range(self.epochs):
                for inputs, labels, batch_idx in self.batch_generator(
                    split="train", desc=f"Conjugate Gradient Training (epoch {epoch_idx})"
                ):
                    self.trained_batches += 1
                    initial_ensemble_weights = self.ensemble_weights.clone()
                    for step_idx in range(max_steps_per_batch):
                        gradient, hessian = self._strategy.get_gradient_and_hessian(inputs, labels)
                        delta_ensemble_weights = self._conjugate_gradient(
                            gradient=gradient,
                            hessian=hessian,
                            initial_guess=initial_guess,
                            stop_threshold=stop_threshold,
                        )
                        # Break if there is a nan (often due to tiny gradients or hessians) or
                        # the update threshold is reached
                        if (
                            delta_ensemble_weights.isnan().any()
                            or delta_ensemble_weights.abs().max() < stop_threshold
                        ):
                            break
                        # Clamp to help prevent runaways due to noise or a poor starting point.
                        delta_ensemble_weights = delta_ensemble_weights.clamp(
                            min=-clip_magnitude,
                            max=clip_magnitude,
                        )
                        self.ensemble_weights += delta_ensemble_weights
                    # Update ensemble_weights with an exponential moving average.
                    self.ensemble_weights = (
                        1 - ema_weight
                    ) * initial_ensemble_weights + ema_weight * self.ensemble_weights
                    preds = self.get_ensembled_preds_from_inputs(inputs)
                    self.update_metrics(preds, labels, split="train")
                    ensemble_weight_dict = {
                        f"ensemble_weight_{idx}": w.item()
                        for idx, w in enumerate(self.ensemble_weights)
                    }
                    if self.is_chief:
                        self.report_metrics(split="train", additional_metrics=ensemble_weight_dict)
                    self.reset_metrics(split="train")
                    ensemble_weight_history.append(self.ensemble_weights.clone())
            if self.core_context.preempt.should_preempt():
                return

    def train_transformer(self) -> None:
        """Trains the ensemble transformer."""
        pass
