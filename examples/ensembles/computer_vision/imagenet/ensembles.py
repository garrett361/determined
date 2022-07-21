import abc
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


class Ensemble(nn.Module):
    def __init__(
        self,
        core_context,
        model_list: List[nn.Module],
        train_batch_size: int,
        val_batch_size: int,
        dataset_name: str,
        ensemble_strategy: str,
        extra_val_log_metrics: Dict[str, Any] = None,
        sanity_check: bool = False,
        num_combinations: Optional[int] = None,
        lr: Optional[float] = None,
        epochs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.core_context = core_context
        self.models = nn.ModuleList(model_list)
        self.num_models = len(self.models)
        self.models.eval()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataset_name = dataset_name
        self.ensemble_strategy = ensemble_strategy
        self.extra_val_log_metrics = extra_val_log_metrics or {}
        self.sanity_check = sanity_check
        if self.sanity_check:
            print(f"Running in sanity check mode!")
        self.num_combinations = num_combinations
        self.lr = lr
        self.epochs = epochs

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
            "naive": NaiveStrategy,
            "naive_temp": NaiveTempStrategy,
            "naive_logits": NaiveLogitsStrategy,
            "naive_logits_temp": NaiveLogitsTempStrategy,
            "most_confident": MostConfidentStrategy,
            "most_confident_temp": MostConfidentTempStrategy,
            "majority_vote": MajorityVoteStrategy,
            "vbmc": VBMCStrategy,
            "vbmc_temp": VBMCTempStrategy,
            "super_learner_probs": SuperLearnerProbsStrategy,
            "super_learner_probs_temp": SuperLearnerProbsTempStrategy,
            "super_learner_logits": SuperLearnerLogitsStrategy,
        }
        self._strategy = self._ensemble_strategies[self.ensemble_strategy](self)

        if self._strategy.requires_training:
            print(f"Building train_dataset")
            self.train_dataset = data.get_dataset(name=self.dataset_name, split="train")
            print(f"{len(self.train_dataset)} records in train_dataset")
        else:
            print(f"Skipping building train_dataset")
            self.train_dataset = None
        print(f"Building val_dataset")
        self.val_dataset = data.get_dataset(name=self.dataset_name, split="val")
        print(f"{len(self.val_dataset)} records in val_dataset")
        self.trained_batches = 0
        self.train_loader = self.build_train_loader()
        self.val_loader = self.build_val_loader()
        self._nll_criterion = nn.NLLLoss()

        # There can be multiple notions of weights for different ensemble strategies.  `weights`
        # are generally used to weight the final individual model probabilities or
        # logits, while _other_weights cover other forms of weights.

        # Some strategies require training the ensemble weights
        self.ensemble_weights = None
        if self._strategy.requires_SGD:
            self.ensemble_weights = nn.Parameter(
                torch.ones(len(self.models), device=self.device, requires_grad=True)
            )
            # We only train the ensemble weights:
            self.optimizer = torch.optim.Adam([self.ensemble_weights], lr=self.lr)
        self._other_weights = None
        # Others need log-likelihoods at intermediate steps.
        self._log_likelihoods = None
        # Sometimes we calibrate using an inverse temperature.
        self.beta = None

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
        self._strategy.build_fn()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns logits for the models, stacked along the last dimension."""
        model_logits = torch.stack([model(inputs) for model in self.models], dim=-1)
        return model_logits

    def get_ensembled_preds_from_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns predictions for the ensembled model."""
        logits = self(inputs)
        ensembled_preds = self._strategy.pred_fn(logits)
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
                probs = self.get_ensembled_preds_from_inputs(inputs)
                self.update_metrics(probs, labels)
            if self.is_chief:
                computed_metrics = self.compute_metrics("val_")
                conflicted_keys = set(self.extra_val_log_metrics) & set(computed_metrics)
                if conflicted_keys:
                    raise ValueError(
                        f"extra_val_log_metrics/val_metrics conflicting keys: {conflicted_keys}"
                    )
                # Join with extra_val_log_metrics and remove any None-valued metrics with a
                # warning (these would throw errors). Also include the weights and _beta.
                reported_metrics = {**self.extra_val_log_metrics, **computed_metrics}
                if self.ensemble_weights is not None:
                    reported_metrics["ensemble_weights"] = [w.item() for w in self.ensemble_weights]
                if self.beta is not None:
                    reported_metrics["beta"] = [b.item() for b in self.beta]
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
            loss = self._nll_criterion(probs.log(), labels)
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

    def train_vbmc(self) -> None:
        """Computes the posterior and resulting effective weights for the num_combination linear
        combinations of models."""
        with torch.no_grad():
            self._log_likelihoods = torch.zeros(self.num_combinations, device=self.device)
            vbmc_criterion = nn.NLLLoss(reduction="none")
            for batch_idx, batch in enumerate(tqdm.tqdm(self.train_loader, desc="Training VBMC")):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
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

            # Prevent overflow for the summed log-likelihoods
            self._log_likelihoods -= self._log_likelihoods.mean()
            overflow_diff = self._log_likelihoods.max() - MAX_EXP_ARG
            if overflow_diff > 0:
                self._log_likelihoods -= overflow_diff
        posterior = self._log_likelihoods.softmax(dim=0)
        self.ensemble_weights = self._other_weights @ posterior
        if self.sanity_check:
            prob_sum_check = self.ensemble_weights.sum(dim=-1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        reported_metrics = {
            "posterior": [p.item() for p in posterior],
            "ensemble_weights": [w.item() for w in self.ensemble_weights],
        }
        self.core_context.train.report_training_metrics(
            steps_completed=self.trained_batches, metrics=reported_metrics
        )
        if self.core_context.preempt.should_preempt():
            return

    def calibrate_temperature(self) -> None:
        """Calibrates temperatures for all base models in the ensemble in parallel using Newton's
        method."""
        self.beta = torch.ones(self.num_models, device=self.device)
        with torch.no_grad():
            beta_history = [self.beta.clone()]
            for batch_idx, batch in enumerate(
                tqdm.tqdm(self.train_loader, desc="Calibrating temperature")
            ):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                score = self(inputs)
                probs = (self.beta * score).softmax(dim=1)
                mean_true_score = score[torch.arange(len(labels)), labels].mean(dim=0)
                score_label_mean = (probs * score).sum(dim=1)
                score2_label_mean = (probs * score ** 2).sum(dim=1)
                dloss_dbeta = score_label_mean.mean(dim=0) - mean_true_score
                dloss_dbeta2 = (score2_label_mean - score_label_mean ** 2).mean(dim=0)
                delta_beta = -1 * dloss_dbeta / dloss_dbeta2
                # Clamp to help prevent runaways due to noise
                delta_beta = delta_beta.clamp(min=-1, max=1)
                self.beta += delta_beta
                beta_history.append(self.beta.clone())
                self.trained_batches += 1
            self.beta = torch.stack(beta_history, dim=0).mean(dim=0)
        reported_metrics = {"betas": [b.item() for b in self.beta]}
        self.core_context.train.report_training_metrics(
            steps_completed=self.trained_batches, metrics=reported_metrics
        )
        if self.core_context.preempt.should_preempt():
            return

    def train_super_learner(self) -> None:
        """Train super-learner strategies using a standard SGD loop."""
        # TODO: Optimize by using CrossEntropyLoss when possible.
        for epoch_idx in tqdm.tqdm(range(self.epochs), desc="Epoch"):
            for batch_idx, batch in enumerate(
                tqdm.tqdm(self.train_loader, desc="Training Super Learner")
            ):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                probs = self.get_ensembled_preds_from_inputs(inputs)
                self.update_metrics(probs, labels)

                self.optimizer.zero_grad()
                loss = self._nll_criterion(probs.log(), labels)
                loss.backward()
                self.optimizer.step()
                self.trained_batches += 1
            # Report training metrics at the end of each epoch
            if self.is_chief:
                reported_metrics = self.compute_metrics("train_")
                reported_metrics["ensemble_weights"] = [w.item() for w in self.ensemble_weights]
                for key in list(reported_metrics):
                    if reported_metrics[key] is None:
                        warnings.warn(f"Removing train metric {key} whose value is None.")
                        reported_metrics.pop(key)

                self.core_context.train.report_validation_metrics(
                    steps_completed=self.trained_batches, metrics=reported_metrics
                )
            self.reset_metrics()
            if self.core_context.preempt.should_preempt():
                return


class Strategy(abc.ABC):
    def __init__(
        self,
        ensemble: Ensemble,
    ) -> None:
        self.ensemble = ensemble

    @property
    @abc.abstractmethod
    def generates_probabilities(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def requires_training(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def requires_SGD(self) -> bool:
        pass

    @abc.abstractmethod
    def build_fn(self) -> None:
        """Performs all setup and training necessary for the strategy."""
        pass

    @abc.abstractmethod
    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns a prediction given the logits of the base models in the ensemble, stacked along
        the final dimension.  The output should either be in the form of probabilities or a
        single prediction."""
        pass


class NaiveStrategy(Strategy):
    """Average the probabilities of all models with equal weights."""

    generates_probabilities = True
    requires_training = False
    requires_SGD = False

    def build_fn(self) -> None:
        self.ensemble.ensemble_weights = (
            torch.ones(self.ensemble.num_models, device=self.ensemble.device)
            / self.ensemble.num_models
        )

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = logits.softmax(dim=1)
        ensemble_prob = model_probs @ self.ensemble.ensemble_weights
        return ensemble_prob


class NaiveTempStrategy(Strategy):
    """Average the probabilities of all models with equal weights, after calibrating the
    temperature.
    """

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build_fn(self) -> None:
        self.ensemble.ensemble_weights = (
            torch.ones(self.ensemble.num_models, device=self.ensemble.device)
            / self.ensemble.num_models
        )
        self.ensemble.calibrate_temperature()

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = (logits * self.ensemble.beta).softmax(dim=1)
        ensemble_prob = model_probs @ self.ensemble.ensemble_weights
        return ensemble_prob


class NaiveLogitsStrategy(Strategy):
    """Average the logits of all models with equal weights."""

    generates_probabilities = True
    requires_training = False
    requires_SGD = False

    def build_fn(self) -> None:
        self.ensemble.ensemble_weights = (
            torch.ones(self.ensemble.num_models, device=self.ensemble.device)
            / self.ensemble.num_models
        )

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        ensemble_logits = logits @ self.ensemble.ensemble_weights
        ensemble_prob = ensemble_logits.softmax(dim=1)
        return ensemble_prob


class NaiveLogitsTempStrategy(Strategy):
    """Average the logits of all models with equal weights after calibrating the temperature."""

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build_fn(self) -> None:
        self.ensemble.ensemble_weights = (
            torch.ones(self.ensemble.num_models, device=self.ensemble.device)
            / self.ensemble.num_models
        )
        self.ensemble.calibrate_temperature()

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        ensemble_logits = (logits * self.ensemble.beta) @ self.ensemble.ensemble_weights
        ensemble_prob = ensemble_logits.softmax(dim=1)
        return ensemble_prob


class MostConfidentStrategy(Strategy):
    """For each sample, use the prediction of the most-confident model."""

    generates_probabilities = True
    requires_training = False
    requires_SGD = False

    def build_fn(self) -> None:
        pass

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        ensembles_probs = logits.softmax(dim=1)
        max_idxs = ensembles_probs.max(dim=1).values.argmax(dim=-1)
        ensemble_prob = ensembles_probs[torch.arange(len(max_idxs)), ..., max_idxs]
        return ensemble_prob


class MostConfidentTempStrategy(Strategy):
    """For each sample, use the prediction of the most-confident model, after calibrating the
    temperature.
    """

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build_fn(self) -> None:
        self.ensemble.calibrate_temperature()

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        ensembles_probs = (logits * self.ensemble.beta).softmax(dim=1)
        max_idxs = ensembles_probs.max(dim=1).values.argmax(dim=-1)
        ensemble_prob = ensembles_probs[torch.arange(len(max_idxs)), ..., max_idxs]
        return ensemble_prob


class MajorityVoteStrategy(Strategy):
    """For each sample, use a majority vote among models. Torch breaks ties by choosing the
    lowest prediction index among tied values.
    """

    generates_probabilities = False
    requires_training = False
    requires_SGD = False

    def build_fn(self) -> None:
        pass

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        ensembles_preds = logits.argmax(dim=1)
        majority_vote_preds = ensembles_preds.mode(-1).values
        return majority_vote_preds


class VBMCStrategy(Strategy):
    """Vectorized Bayesian Model Combination."""

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build_fn(self) -> None:
        # Generate num_combinations sets of model combinations.
        # We use a flat prior over this discrete set.
        dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(self.ensemble.num_models))
        self.ensemble._other_weights = (
            dirichlet.sample((self.ensemble.num_combinations,)).to(self.ensemble.device).T
        )
        # _other_weights is (self.ensemble.num_models, self.ensemble.num_combinations)-shaped
        self.ensemble.train_vbmc()

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        if self.ensemble.sanity_check:
            prob_sum_check = self.ensemble.ensemble_weights.sum(dim=-1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        model_probs = logits.softmax(dim=1)
        ensemble_prob = model_probs @ self.ensemble.ensemble_weights
        return ensemble_prob


class VBMCTempStrategy(Strategy):
    """Vectorized Bayesian Model Combination, with temperature calibration."""

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build_fn(self) -> None:
        # Generate num_combinations sets of model combinations.
        # We use a flat prior over this discrete set.
        dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(self.ensemble.num_models))
        self.ensemble._other_weights = (
            dirichlet.sample((self.ensemble.num_combinations,)).to(self.ensemble.device).T
        )
        # _other_weights is (self.ensemble.num_models, self.ensemble.num_combinations)-shaped
        self.ensemble.calibrate_temperature()
        self.ensemble.train_vbmc()

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        if self.ensemble.sanity_check:
            prob_sum_check = self.ensemble.ensemble_weights.sum(dim=-1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        model_probs = (logits * self.ensemble.beta).softmax(dim=1)
        ensemble_prob = model_probs @ self.ensemble.ensemble_weights
        return ensemble_prob


class SuperLearnerProbsStrategy(Strategy):
    """Minimize the KL divergence for a weighted sum of model probabilities, with the weights
    adding to unity.
    """

    generates_probabilities = True
    requires_training = True
    requires_SGD = True

    def build_fn(self) -> None:
        self.ensemble.train_super_learner()

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = logits.softmax(dim=1)
        ensemble_prob = model_probs @ self.ensemble.ensemble_weights.softmax(dim=-1)
        return ensemble_prob


class SuperLearnerProbsTempStrategy(Strategy):
    """Minimize the KL divergence for a weighted sum of temperature-calibrated model probabilities,
    with the weights adding to unity.
    """

    generates_probabilities = True
    requires_training = True
    requires_SGD = True

    def build_fn(self) -> None:
        self.ensemble.calibrate_temperature()
        self.ensemble.train_super_learner()

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = logits.softmax(dim=1)
        ensemble_prob = model_probs @ self.ensemble.ensemble_weights.softmax(dim=-1)
        return ensemble_prob


class SuperLearnerLogitsStrategy(Strategy):
    """Minimize the KL divergence for a weighted sum of model logits, with no constraint on the
    weights.
    """

    generates_probabilities = True
    requires_training = True
    requires_SGD = True

    def build_fn(self) -> None:
        self.ensemble.train_super_learner()

    def pred_fn(self, logits: torch.Tensor) -> torch.Tensor:
        ensemble_logits = logits @ self.ensemble.ensemble_weights
        ensemble_prob = ensemble_logits.softmax(dim=1)
        return ensemble_prob
