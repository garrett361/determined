from typing import Any

import torch
import torch.nn.functional as F

import torchmetrics


class NLLMeanMetric(torchmetrics.Metric):
    """Simple NLL mean metric which takes in probs, labels pairs."""

    value: torch.Tensor
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("records", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, labels: torch.Tensor) -> None:  # type: ignore
        assert probs.shape[0] == labels.shape[0], "probs and labels must have the same batch size"
        batch_size = probs.shape[0]
        nll_loss = F.nll_loss(probs.log(), labels)

        self.value += nll_loss * batch_size
        self.records += batch_size

    def compute(self) -> torch.Tensor:
        """Compute the aggregated mean NLL loss."""
        return self.value / self.records


# class BatchDimMeanMetric(BaseAggregator):
#     """Computes vectorized means over the batch dimension, assumed to be the leading dimension.
#     E.g., the compute method returns the sum of all passed in tensor values divided by the total
#     number of processed records.
#     """
#
#     def __init__(
#         self,
#         nan_strategy: Union[str, float] = "warn",
#     ):
#         super().__init__(
#             "sum",
#             torch.tensor(0.0),
#             nan_strategy,
#         )
#         self.add_state("records", default=torch.tensor(0.0), dist_reduce_fx="sum")
#
#     def update(self, value: torch.Tensor) -> None:  # type: ignore
#         """Update state with data."""
#         value = self._cast_and_nan_check_input(value)
#
#         if value.numel() == 0:
#             return
#         self.value = self.value + value.sum(0)
#         self.records += value.shape[0]
#
#     def compute(self) -> torch.Tensor:
#         """Compute the aggregated value."""
#         return self.value / self.records
#
#
# class VectorizedMeanMetric(BaseAggregator):
#     """Computes vectorized means, averaged over the number of consumed tensors."""
#
#     def __init__(
#         self,
#         nan_strategy: Union[str, float] = "warn",
#     ):
#         super().__init__(
#             "sum",
#             torch.tensor(0.0),
#             nan_strategy,
#         )
#         self.add_state("updates", default=torch.tensor(0.0), dist_reduce_fx="sum")
#
#     def update(self, value: torch.Tensor) -> None:  # type: ignore
#         """Update state with data."""
#         value = self._cast_and_nan_check_input(value)
#
#         if value.numel() == 0:
#             return
#         self.value = self.value + value
#         self.updates += 1
#
#     def compute(self) -> torch.Tensor:
#         """Compute the aggregated value."""
#         return self.value / self.updates
#
#
# class MultiModelAccuracy(BaseAggregator):
#     """Computes accuracies for multiple models computed in parallel,
#     assuming the final dimension indexes models.
#     """
#
#     def __init__(
#         self,
#         nan_strategy: Union[str, float] = "warn",
#         dim: int = 1,
#         k: int = 1,
#     ):
#         super().__init__(
#             "sum",
#             torch.tensor(0.0),
#             nan_strategy,
#         )
#         self.dim = dim
#         self.k = k
#         self.add_state("records", default=torch.tensor(0.0), dist_reduce_fx="sum")
#
#     def update(self, output: torch.Tensor, labels: torch.Tensor) -> None:  # type: ignore
#         """Update state with data."""
#         output = self._cast_and_nan_check_input(output)
#         labels = self._cast_and_nan_check_input(labels)
#         records = output.shape[0]
#         if output.numel() == 0:
#             return
#         assert (
#             len(output.shape) == 3
#         ), "output should be a 3D tensor, consisting of batch, pred, and model dims, in that order"
#
#         preds = output.topk(k=self.k, dim=1).indices
#         non_model_dims = len(preds.shape) - 1
#         for _ in range(non_model_dims):
#             labels = labels[:, None]
#         correct_preds = labels.expand_as(preds) == preds
#         for _ in range(non_model_dims):
#             correct_preds = correct_preds.sum(0)
#         self.value = self.value + correct_preds
#         self.records += records
#
#     def compute(self) -> torch.Tensor:
#         """Compute the aggregated value."""
#         return self.value / self.records
