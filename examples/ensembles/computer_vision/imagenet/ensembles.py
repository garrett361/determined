import abc
import logging
import random
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, Union

import determined as det
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

logging.basicConfig(level=logging.DEBUG, format=det.LOG_FORMAT)

MAX_EXP_ARG = 88.0


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
        extra_val_log_metrics: Dict[str, Any] = None,
        sanity_check: bool = False,
        num_combinations: Optional[int] = None,
        lr: Optional[float] = None,
        epochs: Optional[int] = None,
        aggregation_batches: Optional[int] = 4,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.core_context = core_context
        self.models = nn.ModuleList(models)
        self.transforms = transforms
        self.num_models = len(self.models)
        self.models.eval()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataset_name = dataset_name
        self.ensemble_strategy = ensemble_strategy
        self.extra_val_log_metrics = extra_val_log_metrics or {}
        self.sanity_check = sanity_check
        if self.sanity_check:
            logging.info(f"Running in sanity check mode!")
        self.num_combinations = num_combinations
        self.lr = lr
        self.epochs = epochs
        self.aggregation_batches = aggregation_batches
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
        self.betas = None

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
        with torch.no_grad():
            for inputs, labels, batch_idx in self.batch_generator(split="val", desc="Validating"):
                labels = labels.to(self.device)
                probs = self.get_ensembled_preds_from_inputs(inputs)
                self.update_metrics(probs, labels)
            if self.is_chief:
                self.report_val_metrics()
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

    def compute_metrics(self, prefix: str = "") -> Dict[str, Any]:
        computed_metrics = {
            prefix + name: metric.compute().item() for name, metric in self.accuracy_metrics.items()
        }
        if self.loss_metric is not None:
            computed_metrics[prefix + "loss"] = self.loss_metric.compute().item()
        return computed_metrics

    def reset_metrics(self) -> None:
        for met in self.accuracy_metrics.values():
            met.reset()
        if self.loss_metric is not None:
            self.loss_metric.reset()

    def report_val_metrics(self) -> None:
        computed_metrics = self.compute_metrics("val_")
        conflicted_keys = set(self.extra_val_log_metrics) & set(computed_metrics)
        if conflicted_keys:
            raise ValueError(
                f"extra_val_log_metrics/val_metrics conflicting keys: {conflicted_keys}"
            )
        # Join with extra_val_log_metrics and remove any None-valued metrics with a
        # warning (these would throw errors). Also include the weights and betas.
        reported_metrics = {**self.extra_val_log_metrics, **computed_metrics}
        if self.ensemble_weights is not None:
            reported_metrics["ensemble_weights"] = [w.item() for w in self.ensemble_weights]
        if self.betas is not None:
            reported_metrics["betas"] = [b.item() for b in self.betas]
        for key in list(reported_metrics):
            if reported_metrics[key] is None:
                logging.warning(f"Removing val metric {key} whose value is None.")
                reported_metrics.pop(key)

        self.core_context.train.report_validation_metrics(
            steps_completed=self.trained_batches, metrics=reported_metrics
        )

    def report_train_metrics(
        self, extra_train_log_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        extra_train_log_metrics = extra_train_log_metrics or {}
        computed_metrics = self.compute_metrics("train_")
        conflicted_keys = set(extra_train_log_metrics) & set(computed_metrics)
        if conflicted_keys:
            raise ValueError(
                f"extra_train_log_metrics/train_metrics conflicting keys: {conflicted_keys}"
            )
        reported_metrics = {**extra_train_log_metrics, **computed_metrics}
        self.core_context.train.report_training_metrics(
            steps_completed=self.trained_batches, metrics=reported_metrics
        )

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

    def calibrate_temperature(
        self,
        stop_threshold: float = 1e-4,
        max_steps_per_batch: int = 5,
        clip_magnitude: float = 0.1,
        ewa_weight: float = 0.95,
    ) -> None:
        """Calibrates temperatures for all base models in the ensemble in parallel using Newton's
        method."""
        self.betas = torch.ones(self.num_models, device=self.device)
        with torch.no_grad():
            beta_history = [self.betas.clone()]
            for epoch_idx in range(self.epochs):
                for inputs, labels, batch_idx in self.batch_generator(
                    split="train", desc=f"Calibrating Temperature (epoch {epoch_idx})"
                ):
                    self.trained_batches += 1
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
                    beta_history.append(self.betas.clone())
                    beta_dict = {f"beta_{idx}": b.item() for idx, b in enumerate(self.betas)}
                    if self.is_chief:
                        self.core_context.train.report_training_metrics(
                            steps_completed=self.trained_batches, metrics=beta_dict
                        )
            # Use an exponential-weighted average of the final weights for each batch as the
            # ultimate ensemble weights.
            stacked_beta_history = torch.stack(beta_history, dim=-1)
            history_len = stacked_beta_history.shape[-1]
            ewa_weights = torch.tensor(
                [ewa_weight ** n for n in reversed(range(history_len))], device=self.device
            )
            ewa_weights = ewa_weights / ewa_weights.sum()
            self.betas = stacked_beta_history @ ewa_weights
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
        clip_magnitude: float = 0.1,
        ewa_weight: float = 0.95,
    ) -> None:
        with torch.no_grad():
            ensemble_weight_history = [self.ensemble_weights.clone()]
            for epoch_idx in range(self.epochs):
                for inputs, labels, batch_idx in self.batch_generator(
                    split="train", desc=f"Conjugate Gradient Training (epoch {epoch_idx})"
                ):
                    self.trained_batches += 1
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
                    preds = self.get_ensembled_preds_from_inputs(inputs)
                    self.update_metrics(preds, labels)
                    ensemble_weight_dict = {
                        f"ensemble_weight_{idx}": w.item()
                        for idx, w in enumerate(self.ensemble_weights)
                    }
                    if self.is_chief:
                        self.report_train_metrics(extra_train_log_metrics=ensemble_weight_dict)
                    self.reset_metrics()
                    ensemble_weight_history.append(self.ensemble_weights.clone())
            # Use an exponential-weighted average of the final weights for each batch as the
            # ultimate ensemble weights.
            stacked_ensemble_weight_history = torch.stack(ensemble_weight_history, dim=-1)
            history_len = stacked_ensemble_weight_history.shape[-1]
            ewa_weights = torch.tensor(
                [ewa_weight ** n for n in reversed(range(history_len))], device=self.device
            )
            ewa_weights = ewa_weights / ewa_weights.sum()
            self.ensemble_weights = stacked_ensemble_weight_history @ ewa_weights
            if self.core_context.preempt.should_preempt():
                return


class Strategy(abc.ABC):
    def __init__(
        self,
        ensemble: ClassificationEnsemble,
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
    def build(self) -> None:
        """Performs all setup and training necessary for the strategy."""
        pass

    @abc.abstractmethod
    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns a prediction given the logits of the base models in the ensemble, stacked along
        the final dimension.  The output should either be in the form of probabilities or a
        single prediction."""
        pass

    def get_gradient_and_hessian(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the relevant gradient and hessian. For use, e.g., with the conjugate gradient
        method.
        """
        pass

    def _initialize_uniform_ensemble_weights(self) -> None:
        self.ensemble.ensemble_weights = (
            torch.ones(self.ensemble.num_models, device=self.ensemble.device)
            / self.ensemble.num_models
        )


class NaiveStrategy(Strategy):
    """Average the probabilities of all models with equal weights."""

    generates_probabilities = True
    requires_training = False
    requires_SGD = False

    def build(self) -> None:
        self._initialize_uniform_ensemble_weights()

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = logits.softmax(dim=1)
        ensemble_probs = model_probs @ self.ensemble.ensemble_weights
        return ensemble_probs


class NaiveTempStrategy(Strategy):
    """Average the probabilities of all models with equal weights, after calibrating the
    temperature.
    """

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build(self) -> None:
        self._initialize_uniform_ensemble_weights()
        self.ensemble.calibrate_temperature()

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = (logits * self.ensemble.betas).softmax(dim=1)
        ensemble_probs = model_probs @ self.ensemble.ensemble_weights
        return ensemble_probs


class NaiveLogitsStrategy(Strategy):
    """Average the logits of all models with equal weights."""

    generates_probabilities = True
    requires_training = False
    requires_SGD = False

    def build(self) -> None:
        self._initialize_uniform_ensemble_weights()

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        ensemble_logits = logits @ self.ensemble.ensemble_weights
        ensemble_probs = ensemble_logits.softmax(dim=1)
        return ensemble_probs


class NaiveLogitsTempStrategy(Strategy):
    """Average the logits of all models with equal weights after calibrating the temperature."""

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build(self) -> None:
        self._initialize_uniform_ensemble_weights()
        self.ensemble.calibrate_temperature()

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        ensemble_logits = (logits * self.ensemble.betas) @ self.ensemble.ensemble_weights
        ensemble_probs = ensemble_logits.softmax(dim=1)
        return ensemble_probs


class MostConfidentStrategy(Strategy):
    """For each sample, use the prediction of the most-confident model."""

    generates_probabilities = True
    requires_training = False
    requires_SGD = False

    def build(self) -> None:
        pass

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = logits.softmax(dim=1)
        max_idxs = model_probs.max(dim=1).values.argmax(dim=-1)
        ensemble_probs = model_probs[torch.arange(len(max_idxs)), ..., max_idxs]
        return ensemble_probs


class MostConfidentTempStrategy(Strategy):
    """For each sample, use the prediction of the most-confident model, after calibrating the
    temperature.
    """

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build(self) -> None:
        self.ensemble.calibrate_temperature()

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = (logits * self.ensemble.betas).softmax(dim=1)
        max_idxs = model_probs.max(dim=1).values.argmax(dim=-1)
        ensemble_probs = model_probs[torch.arange(len(max_idxs)), ..., max_idxs]
        return ensemble_probs


class MajorityVoteStrategy(Strategy):
    """For each sample, use a majority vote among models. Torch breaks ties by choosing the
    lowest prediction index among tied values.
    """

    generates_probabilities = False
    requires_training = False
    requires_SGD = False

    def build(self) -> None:
        pass

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        ensembles_preds = logits.argmax(dim=1)
        majority_vote_preds = ensembles_preds.mode(-1).values
        return majority_vote_preds


class VBMCStrategy(Strategy):
    """Vectorized Bayesian Model Combination."""

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build(self) -> None:
        # Generate num_combinations sets of model combinations.
        # We use a flat prior over this discrete set.
        dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(self.ensemble.num_models))
        self.ensemble._other_weights = (
            dirichlet.sample((self.ensemble.num_combinations,)).to(self.ensemble.device).T
        )
        # _other_weights is (self.ensemble.num_models, self.ensemble.num_combinations)-shaped
        self.ensemble.train_vbmc()

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        if self.ensemble.sanity_check:
            prob_sum_check = self.ensemble.ensemble_weights.sum(dim=-1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        model_probs = logits.softmax(dim=1)
        ensemble_probs = model_probs @ self.ensemble.ensemble_weights
        return ensemble_probs


class VBMCTempStrategy(Strategy):
    """Vectorized Bayesian Model Combination, with temperature calibration."""

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build(self) -> None:
        # Generate num_combinations sets of model combinations.
        # We use a flat prior over this discrete set.
        dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(self.ensemble.num_models))
        self.ensemble._other_weights = (
            dirichlet.sample((self.ensemble.num_combinations,)).to(self.ensemble.device).T
        )
        # _other_weights is (self.ensemble.num_models, self.ensemble.num_combinations)-shaped
        self.ensemble.calibrate_temperature()
        self.ensemble.train_vbmc()

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        if self.ensemble.sanity_check:
            prob_sum_check = self.ensemble.ensemble_weights.sum(dim=-1)
            torch.testing.assert_close(prob_sum_check, torch.ones_like(prob_sum_check))
        model_probs = (logits * self.ensemble.betas).softmax(dim=1)
        ensemble_probs = model_probs @ self.ensemble.ensemble_weights
        return ensemble_probs


class SuperLearnerProbsStrategy(Strategy):
    """Minimize the KL divergence for a weighted sum of model probabilities, with the weights
    adding to unity.
    """

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build(self) -> None:
        self._initialize_uniform_ensemble_weights()
        self.ensemble.train_ensemble_weights_with_conjugate_gradient()

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = logits.softmax(dim=1)
        ensemble_probs = model_probs @ self.ensemble.ensemble_weights.softmax(dim=-1)
        return ensemble_probs

    def get_gradient_and_hessian(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_probs = self.ensemble(inputs).softmax(dim=-1)
        model_probs_star = model_probs[torch.arange(len(labels)), labels]
        ensemble_weight_probs = self.ensemble.ensemble_weights.softmax(dim=-1)
        ensemble_probs = model_probs @ ensemble_weight_probs
        ensemble_probs_star = ensemble_probs[torch.arange(len(labels)), labels]

        model_div_ensemble_probs_data_mean = (
            model_probs_star / ensemble_probs_star[..., None]
        ).mean(dim=0)

        gradient = ensemble_weight_probs * (1 - model_div_ensemble_probs_data_mean)

        hessian_diagonal_term = (
            torch.eye(len(ensemble_weight_probs), device=self.ensemble.device)
            * ensemble_weight_probs
            * (1 - model_div_ensemble_probs_data_mean)
        )
        hessian_remainder_data_mean_term = (
            model_probs_star[:, None]
            * model_probs_star[..., None]
            / ensemble_probs_star[..., None, None] ** 2
        ).mean(dim=0)
        hessian_remainder = (
            ensemble_weight_probs[None, ...]
            * ensemble_weight_probs[..., None]
            * hessian_remainder_data_mean_term
        )
        hessian = hessian_diagonal_term + hessian_remainder

        if self.ensemble.sanity_check:
            torch.testing.assert_close(
                gradient.sum(), torch.tensor(0.0, device=self.ensemble.device)
            )
            torch.testing.assert_close(
                hessian.sum(), torch.tensor(1.0, device=self.ensemble.device)
            )

        return gradient, hessian


class SuperLearnerProbsTempStrategy(SuperLearnerProbsStrategy):
    """Minimize the KL divergence for a weighted sum of temperature-calibrated model probabilities,
    with the weights adding to unity.
    """

    def build(self) -> None:
        self._initialize_uniform_ensemble_weights()
        self.ensemble.calibrate_temperature()
        self.ensemble.train_ensemble_weights_with_conjugate_gradient()


class SuperLearnerLogitsStrategy(Strategy):
    """Minimize the KL divergence for a weighted sum of model logits, with no constraint on the
    weights.
    """

    generates_probabilities = True
    requires_training = True
    requires_SGD = False

    def build(self) -> None:
        self._initialize_uniform_ensemble_weights()
        self.ensemble.train_ensemble_weights_with_conjugate_gradient()

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        ensemble_logits = logits @ self.ensemble.ensemble_weights
        ensemble_probs = ensemble_logits.softmax(dim=1)
        return ensemble_probs

    def get_gradient_and_hessian(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.ensemble(inputs)
        logits_star = logits[torch.arange(len(labels)), labels]
        ensemble_probs = (logits @ self.ensemble.ensemble_weights).softmax(dim=1)

        logits_star_data_mean = logits_star.mean(dim=0)
        logits_class_mean = (logits * ensemble_probs[..., None]).sum(dim=1)
        logits_class_and_data_mean = logits_class_mean.mean(dim=0)
        gradient = logits_class_and_data_mean - logits_star_data_mean

        logits_12_class_and_data_mean = (
            (logits[..., None] * logits[:, :, None] * ensemble_probs[..., None, None])
            .sum(dim=1)
            .mean(dim=0)
        )
        logits_12_class_then_data_mean = (
            logits_class_mean[:, None] * logits_class_mean[..., None]
        ).mean(dim=0)
        hessian = logits_12_class_and_data_mean - logits_12_class_then_data_mean

        return gradient, hessian
