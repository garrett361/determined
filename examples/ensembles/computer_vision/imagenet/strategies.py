import abc
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class Strategy(abc.ABC):
    def __init__(
        self,
        ensemble: Optional[nn.Module] = None,
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
    def requires_sgd(self) -> bool:
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
        self, inputs: List[torch.Tensor], labels: torch.Tensor
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
    requires_sgd = False

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
    requires_sgd = False

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
    requires_sgd = False

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
    requires_sgd = False

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
    requires_sgd = False

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
    requires_sgd = False

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
    requires_sgd = False

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
    requires_sgd = False

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
    requires_sgd = False

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
    requires_sgd = False

    def build(self) -> None:
        self._initialize_uniform_ensemble_weights()
        self.ensemble.train_ensemble_weights_with_conjugate_gradient()

    def pred(self, logits: torch.Tensor) -> torch.Tensor:
        model_probs = logits.softmax(dim=1)
        ensemble_probs = model_probs @ self.ensemble.ensemble_weights.softmax(dim=-1)
        return ensemble_probs

    def get_gradient_and_hessian(
        self, inputs: List[torch.Tensor], labels: torch.Tensor
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
    requires_sgd = False

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


STRATEGY_DICT = {
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
