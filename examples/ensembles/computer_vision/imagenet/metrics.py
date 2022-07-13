from typing import Any, Union

import torch
from torch import Tensor

from torchmetrics.aggregation import BaseAggregator


class ZeroDimMeanMetric(BaseAggregator):
    """Computes simple, vectorized means, with the mean taken only over
    the zero dimension.
    """

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ):
        super().__init__(
            "sum",
            torch.tensor(0.0),
            nan_strategy,
            **kwargs,
        )
        self.add_state("records", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, value: Union[float, Tensor]) -> None:  # type: ignore
        """Update state with data.
        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
            weight: Either a float or tensor containing weights for calculating
                the average. Shape of weight should be able to broadcast with
                the shape of `value`. Default to `1.0` corresponding to simple
                harmonic average.
        """
        value = self._cast_and_nan_check_input(value)

        if value.numel() == 0:
            return
        self.value = self.value + value.sum(0)
        self.records += value.shape[0]

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return self.value / self.records


class MultiModelAccuracy(BaseAggregator):
    """Computes accuracies for multiple models computed in parallel,
    assuming the final dimension indexes models.
    """

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        dim: int = 1,
        k: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            "sum",
            torch.tensor(0.0),
            nan_strategy,
            **kwargs,
        )
        self.dim = dim
        self.k = k
        self.add_state("records", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, output: Tensor, labels: Tensor) -> None:  # type: ignore
        """Update state with data.
        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
            weight: Either a float or tensor containing weights for calculating
                the average. Shape of weight should be able to broadcast with
                the shape of `value`. Default to `1.0` corresponding to simple
                harmonic average.
        """
        output = self._cast_and_nan_check_input(output)
        labels = self._cast_and_nan_check_input(labels)
        records = output.shape[0]
        if output.numel() == 0:
            return

        preds = output.topk(k=self.k, dim=1).indices
        non_model_dims = len(preds.shape) - 1
        for _ in range(non_model_dims):
            labels = labels[:, None]
        correct_preds = labels.expand_as(preds) == preds
        for _ in range(non_model_dims):
            correct_preds = correct_preds.sum(0)
        self.value = self.value + correct_preds
        self.records += records

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return self.value / self.records
