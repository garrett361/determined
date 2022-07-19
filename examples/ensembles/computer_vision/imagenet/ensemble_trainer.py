from typing import Any, Callable, Dict, List, Optional
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchmetrics
import tqdm

import data

MAX_EXP_ARG = 88.0


class EnsembleStrategy:
    """Small class for storing the build and probability function for an ensemble strategy.
    The pred_fn takes in the vectorized model logits, with the model dimension last, and returns
    the prediction generated by the entire ensemble.
    """

    def __init__(
        self,
        build_fn: Callable,
        pred_fn: Callable,
        generates_probabilities: bool,
        requires_training: bool,
    ) -> None:
        self.build_fn = build_fn
        self.pred_fn = pred_fn
        self.generates_probabilities = generates_probabilities
        self.requires_training = requires_training


class EnsembleTrainer(nn.Module):
    def __init__(
        self,
        core_context,
        model_list: List[nn.Module],
        train_batch_size: int,
        val_batch_size: int,
        dataset_name: str,
        ensemble_strategy: str = "naive",
        ensemble_args: Optional[dict] = None,
        extra_val_log_metrics: Dict[str, Any] = None,
        sanity_check: bool = False,
        num_combinations: Optional[int] = None,
        lr: float = 0.001,
    ) -> None:
        super().__init__()
        self.core_context = core_context
        self.models = nn.ModuleList(model_list)
        self.models.eval()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataset_name = dataset_name
        self.ensemble_strategy = ensemble_strategy
        self.ensemble_args = ensemble_args or {}
        self.extra_val_log_metrics = extra_val_log_metrics or {}
        self.sanity_check = sanity_check
        if self.sanity_check:
            print(f"Running in sanity check mode!")
        self.num_combinations = num_combinations
        self.lr = lr

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

        self._ensemble_strategies = {
            "naive": EnsembleStrategy(
                build_fn=self._build_naive,
                pred_fn=self._naive_pred_fn,
                generates_probabilities=True,
                requires_training=False,
            ),
            "naive_temp": EnsembleStrategy(
                build_fn=self._build_naive_temp,
                pred_fn=self._naive_temp_pred_fn,
                generates_probabilities=True,
                requires_training=True,
            ),
            "naive_logits": EnsembleStrategy(
                build_fn=self._build_naive_logits,
                pred_fn=self._naive_logits_pred_fn,
                generates_probabilities=True,
                requires_training=False,
            ),
            "most_confident": EnsembleStrategy(
                build_fn=self._build_most_confident,
                pred_fn=self._most_confident_pred_fn,
                generates_probabilities=True,
                requires_training=False,
            ),
            "most_confident_temp": EnsembleStrategy(
                build_fn=self._build_most_confident_temp,
                pred_fn=self._most_confident_temp_pred_fn,
                generates_probabilities=True,
                requires_training=True,
            ),
            "majority_vote": EnsembleStrategy(
                build_fn=self._build_majority_vote,
                pred_fn=self._majority_vote_pred_fn,
                generates_probabilities=False,
                requires_training=False,
            ),
            "vbmc": EnsembleStrategy(
                build_fn=self._build_vbmc,
                pred_fn=self._vbmc_pred_fn,
                generates_probabilities=True,
                requires_training=True,
            ),
        }
        self._strategy = self._ensemble_strategies[self.ensemble_strategy]

        if self._strategy.requires_training:
            print(f"Building train_dataset")
            self.train_dataset = data.get_dataset(name=self.dataset_name, split="train")
        else:
            print(f"Skipping building train_dataset")
            self.train_dataset = None
        print(f"Building val_dataset")
        self.val_dataset = data.get_dataset(name=self.dataset_name, split="val")
        self.trained_batches = 0
        self.train_loader = self.build_train_loader()
        self.val_loader = self.build_val_loader()
        self.criterion = nn.NLLLoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=self.lr)

        # There can be multiple notions of weights for different ensemble strategies.  We use
        # _ensemble_weights to give the final probability generated by the whole ensemble, as in
        # self(inputs).sofmax(dim=1) @ self._ensemble_weights, while _model_weights other forms of
        # weights.
        self._ensemble_weights = None
        self._model_weights = None
        # Others need log-likelihoods at intermediate steps.
        self._log_likelihoods = None
        # Sometimes we calibrate using an inverse temperature.
        self._beta = None

        if self._strategy.generates_probabilities:
            self.accuracy_metrics = {
                f"top{k}_acc": torchmetrics.Accuracy(top_k=k) for k in range(1, 11)
            }
        else:
            self.accuracy_metrics = {"top1_acc": torchmetrics.Accuracy()}

        for met in self.accuracy_metrics.values():
            met.to(self.device)

        if self._strategy.generates_probabilities:
            self.loss_metric = torchmetrics.MeanMetric()
            self.loss_metric.to(self.device)
        else:
            self.loss_metric = None

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
        self._ensemble_strategies[self.ensemble_strategy].build_fn(*self.ensemble_args)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns logits for the models, stacked along the last dimension."""
        model_logits = torch.stack([model(inputs) for model in self.models], dim=-1)
        return model_logits

    def get_ensembled_preds(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns predictions for the ensembled model."""
        logits = self(inputs)
        ensembled_preds = self._ensemble_strategies[self.ensemble_strategy].pred_fn(logits)
        if self._strategy.generates_probabilities and self.sanity_check:
            prob_sum_check = ensembled_preds.sum(dim=1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        return ensembled_preds

    def validate_ensemble(self) -> None:
        self.models.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(self.val_loader, desc="Validating")):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                probs = self.get_ensembled_preds(inputs)
                self.update_metrics(probs, labels)
            if self.is_chief:
                computed_metrics = self.compute_metrics("val_")
                conflicted_keys = set(self.extra_val_log_metrics) & set(computed_metrics)
                if conflicted_keys:
                    raise ValueError(
                        f"extra_val_log_metrics/val_metrics conflicting keys: {conflicted_keys}"
                    )
                # Join with extra_val_log_metrics and remove any None-valued metrics with a
                # warning (these would throw errors). Also include the _ensemble_weights and _beta.
                reported_metrics = {**self.extra_val_log_metrics, **computed_metrics}
                reported_metrics["ensemble_weights"] = self._ensemble_weights
                reported_metrics["beta"] = self._beta
                for key in list(reported_metrics):
                    if reported_metrics[key] is None:
                        warnings.warn(f"Removing val metric {key} whose value is None.")
                        reported_metrics.pop(key)

                self.core_context.train.report_validation_metrics(
                    steps_completed=self.trained_batches, metrics=reported_metrics
                )
            self.reset_metrics()
        if self.core_context.preempt.should_preempt():
            return

    def update_metrics(self, probs, labels) -> None:
        for met in self.accuracy_metrics.values():
            met(probs, labels)
        if self.loss_metric is not None:
            # NLLLoss expects log-probabilities
            loss = self.criterion(probs.log(), labels)
            self.loss_metric(loss)

    def compute_metrics(self, prefix: str = ""):
        computed_metrics = {
            prefix + name: metric.compute().item() for name, metric in self.accuracy_metrics.items()
        }
        if self.loss_metric is not None:
            computed_metrics[prefix + "loss"] = self.loss_metric.compute().item()
        return computed_metrics

    def reset_metrics(self):
        for met in self.accuracy_metrics.values():
            met.reset()
        if self.loss_metric is not None:
            self.loss_metric.reset()

    def _build_naive(self) -> None:
        """Average the probabilities of all models with equal weights."""
        num_models = len(self.models)
        self._ensemble_weights = torch.ones(num_models, device=self.device) / num_models

    def _naive_pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = logits.softmax(dim=1)
        ensemble_prob = model_probs @ self._ensemble_weights
        return ensemble_prob

    def _build_naive_logits(self) -> None:
        """Average the logits of all models with equal weights."""
        num_models = len(self.models)
        self._ensemble_weights = torch.ones(num_models, device=self.device) / num_models

    def _naive_logits_pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        ensemble_logits = logits @ self._ensemble_weights
        ensemble_prob = ensemble_logits.softmax(dim=1)
        return ensemble_prob

    def _build_most_confident(self) -> None:
        """For each sample, use the prediction of the most-confident model."""
        pass

    def _most_confident_pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        ensembles_probs = logits.softmax(dim=1)
        max_idxs = ensembles_probs.max(dim=1).values.argmax(dim=-1)
        ensemble_prob = ensembles_probs[torch.arange(len(max_idxs)), ..., max_idxs]
        return ensemble_prob

    def _build_majority_vote(self) -> None:
        """For each sample, use a majority vote among models. Torch breaks ties by choosing the
        lowest prediction index among tied values.
        """
        pass

    def _majority_vote_pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        ensembles_preds = logits.argmax(dim=1)
        majority_vote_preds = ensembles_preds.mode(-1).values
        return majority_vote_preds

    def _build_vbmc(self) -> None:
        """Vectorized Bayesian Model Combination."""
        # Generate num_combinations sets of model combinations. We use a flat prior over this discrete set.
        dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(len(self.models)))
        self._model_weights = dirichlet.sample((self.num_combinations,)).to(self.device).T
        # _model_weights is (len(self.models), self.num_combinations)-shaped
        self._train_vbmc()

    def _vbmc_pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        if self.sanity_check:
            prob_sum_check = self._ensemble_weights.sum(dim=-1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        model_probs = logits.softmax(dim=1)
        ensemble_prob = model_probs @ self._ensemble_weights
        return ensemble_prob

    def _build_vbmc_temp(self) -> None:
        """Vectorized Bayesian Model Combination, with temperature calibration."""
        # Generate num_combinations sets of model combinations. We use a flat prior over this discrete set.
        dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(len(self.models)))
        self._model_weights = dirichlet.sample((self.num_combinations,)).to(self.device).T
        # _model_weights is (len(self.models), self.num_combinations)-shaped
        self._calibrate_temperature()
        self._train_vbmc()

    def _vbmc_temp_pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        if self.sanity_check:
            prob_sum_check = self._ensemble_weights.sum(dim=-1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        model_probs = (logits * self._beta).softmax(dim=1)
        ensemble_prob = model_probs @ self._ensemble_weights
        return ensemble_prob

    def _build_naive_temp(self) -> None:
        """Average the probabilities of all models with equal weights, after calibrating the
        temperature.
        """
        num_models = len(self.models)
        self._ensemble_weights = torch.ones(num_models, device=self.device) / num_models
        self._calibrate_temperature()

    def _naive_temp_pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = (logits * self._beta).softmax(dim=1)
        ensemble_prob = model_probs @ self._ensemble_weights
        return ensemble_prob

    def _build_most_confident_temp(self) -> None:
        """For each sample, use the prediction of the most-confident model, after calibrating the
        temperature.
        """
        self._calibrate_temperature()

    def _most_confident_temp_pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        ensembles_probs = (logits * self._beta).softmax(dim=1)
        max_idxs = ensembles_probs.max(dim=1).values.argmax(dim=-1)
        ensemble_prob = ensembles_probs[torch.arange(len(max_idxs)), ..., max_idxs]
        return ensemble_prob

    def _train_vbmc(self) -> None:
        with torch.no_grad():
            self._log_likelihoods = torch.zeros(self.num_combinations, device=self.device)
            vbmc_criterion = nn.NLLLoss(reduction="none")
            for batch_idx, batch in enumerate(tqdm.tqdm(self.train_loader, desc="Training VBMC")):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self(inputs)
                probs = logits.softmax(dim=1)
                ensembled_probs = probs @ self._model_weights
                log_ensembled_probs = ensembled_probs.log()
                labels = labels[..., None]
                labels = labels.expand(-1, self.num_combinations)
                loss = vbmc_criterion(log_ensembled_probs, labels).sum(dim=0)
                # Subtract loss to get *positive* log-likelihood sum.
                self._log_likelihoods -= loss
                self.trained_batches += 1

            # Prevent overflow for the summed log-likelihoods
            self._log_likelihoods -= self._log_likelihoods.mean()
            overflow_diff = self._log_likelihoods.max() - MAX_EXP_ARG
            if overflow_diff > 0:
                self._log_likelihoods -= overflow_diff
        posterior = self._log_likelihoods.softmax(dim=0)
        self._ensemble_weights = self._model_weights @ posterior
        if self.sanity_check:
            prob_sum_check = self._ensemble_weights.sum(dim=-1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        reported_metrics = {
            "posterior": [p.item() for p in posterior],
            "ensemble_weights": [w.item() for w in self._ensemble_weights],
        }
        self.core_context.train.report_training_metrics(
            steps_completed=self.trained_batches, metrics=reported_metrics
        )
        if self.core_context.preempt.should_preempt():
            return

    def _calibrate_temperature(self) -> None:
        self._beta = torch.ones(len(self.models), device=self.device)
        with torch.no_grad():
            beta_history = [self._beta.clone()]
            for batch_idx, batch in enumerate(
                tqdm.tqdm(self.train_loader, desc="Calibrating temperature")
            ):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                score = self(inputs)
                probs = (self._beta * score).softmax(dim=1)
                mean_true_score = score[torch.arange(len(labels)), labels].mean(dim=0)
                score_label_mean = (probs * score).sum(dim=1)
                score2_label_mean = (probs * score ** 2).sum(dim=1)
                dloss_dbeta = score_label_mean.mean(dim=0) - mean_true_score
                dloss_dbeta2 = (score2_label_mean - score_label_mean ** 2).mean(dim=0)
                delta_beta = -1 * dloss_dbeta / dloss_dbeta2
                # Clamp to help prevent runaways due to noise
                delta_beta = delta_beta.clamp(min=-1, max=1)
                self._beta += delta_beta
                beta_history.append(self._beta.clone())
                self.trained_batches += 1
            self._beta = torch.stack(beta_history, dim=0).mean(dim=0)
        reported_metrics = {"betas": [b.item() for b in self._beta]}
        self.core_context.train.report_training_metrics(
            steps_completed=self.trained_batches, metrics=reported_metrics
        )
        if self.core_context.preempt.should_preempt():
            return
