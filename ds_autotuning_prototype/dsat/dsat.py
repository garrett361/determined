import uuid
from typing import Any, Callable, Dict, List

from determined import searcher

#
# class DSATSearchMethod(searcher.SearchMethod):
#     def __init__(
#         self,
#         max_length: int,
#         max_trials: int,
#         max_concurrent_trials: int = 0,
#     ) -> None:
#         hparams = {"dims": 128, "layers": 1, "num_records": 1000}
#
#         create = searcher.Create(
#             request_id=uuid.uuid4(),
#             hparams=self.search_space(),
#             checkpoint=None,
#         )
#
#         return [create]
#
#     def initial_operations(self, _: searcher.SearcherState) -> List[searcher.Operation]:
#         ops = []
#         create = searcher.Create(
#             request_id=uuid.uuid4(),
#             hparams=self.search_space(),
#             checkpoint=None,
#         )
#         ops.append(create)
#
#         return ops
#
#     def on_trial_created(
#         self, searcher_state: searcher.SearcherState, request_id: uuid.UUID
#     ) -> List[searcher.Operation]:
#         """
#         Informs the searcher that a trial has been created
#         as a result of Create operation.
#         """
#         return []
#
#     def on_validation_completed(
#         self,
#         searcher_state: searcher.SearcherState,
#         request_id: uuid.UUID,
#         metric: float,
#         train_length: int,
#     ) -> List[searcher.Operation]:
#         """
#         Informs the searcher that the validation workload
#         initiated by the same searcher has completed after training for ``train_length`` units.
#         It returns any new operations as a result of this workload completing.
#         """
#         return []
#
#     def on_trial_closed(
#         self, searcher_state: searcher.SearcherState, request_id: uuid.UUID
#     ) -> List[searcher.Operation]:
#         """
#         Informs the searcher that a trial has been closed as a result of a Close
#         operation.
#         """
#         return []
#
#     def progress(self, searcher_state: searcher.SearcherState) -> float:
#         """
#         Returns experiment progress as a float between 0 and 1.
#         """
#         return 0.0
#
#     def on_trial_exited_early(
#         self,
#         searcher_state: searcher.SearcherState,
#         request_id: uuid.UUID,
#         exited_reason: ExitedReason,
#     ) -> List[searcher.Operation]:
#         """
#         Informs the searcher that a trial has exited earlier than expected.
#         """
#         return []
