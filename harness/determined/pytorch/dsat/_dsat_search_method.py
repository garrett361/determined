import argparse
import copy
import json
import logging
import pathlib
import pickle
import random
import uuid
from abc import abstractmethod
from collections import deque
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

from determined import searcher
from determined.experimental.client import create_experiment
from determined.pytorch.dsat import _defaults, _utils
from determined.util import merge_dicts

"""
TODOs:
    * Make sure we don't draw the same config twice in random search.
    * Give control over random seeds and checkpoint rng states.
    * Allow users to configure concurrent trials, somehow.
    * Move away from native DS AT autotuning config syntax, since we're not actually using it as
    they do and there's no reason to constrict ourselves in that way.
    * Clean up directory/import structure. Imports are onerously long now.
    * Don't use stage 3 if stages 1 or 2 work
    * Make it easy for users to subclass the base searcher and use it with dsat.autotune? Not sure
    how we'd do that.
"""


class DSATTrial:
    """
    Helper class for tracking the results and properties of individual Trials.
    """

    def __init__(
        self,
        hparams: Dict[str, Any],
        model_dir: str,
        slots_per_trial: int,
        length: int,
        request_id: Optional[uuid.UUID] = None,
        parent: Optional["DSATTrial"] = None,
        search_data: Optional[Any] = None,
        searcher_metric_name: Optional[str] = None,
    ) -> None:
        self.hparams = hparams
        self.model_dir = model_dir
        self.slots_per_trial = slots_per_trial
        self.length = length
        self.request_id = request_id or uuid.uuid4()
        self.parent = parent
        # Arbitrary attribute for search-specific data tracking.
        self.search_data = search_data
        self.searcher_metric_name = searcher_metric_name

        # Other attrs which are updated during training:
        self.metric = {}
        self.error = False
        self.closed = False
        self.running = False
        self.children = set()

        # If a parent was specified, register the current Trial as the parent's child.
        if self.parent is not None:
            self.parent.children.add(self)

        self.lineage_root = self if self.parent is None else self.parent.lineage_root

        self.ds_config = _utils.get_ds_config_from_hparams(self.hparams, self.model_dir)

        self._error_in_direct_history = False

    @property
    def lineage_set(self) -> Set["DSATTrial"]:
        """Computes set of trials in lineage tree."""
        root = self.lineage_root
        trials_set = {root}
        children = {c for c in root.children}
        while children:
            random_child = children.pop()
            trials_set.add(random_child)
            children |= random_child.children
        return trials_set

    @property
    def num_trials_in_lineage(self) -> int:
        """Computes total number of trials in lineage tree."""
        num_trials = len(self.lineage_set)
        return num_trials

    @property
    def error_in_direct_history(self) -> bool:
        if self._error_in_direct_history:
            return self._error_in_direct_history
        trial = self
        while trial is not None:
            if trial.error:
                return True
            trial = trial.parent
        return False

    @property
    def mbs_in_lineage(self) -> Set["DSATTrial"]:
        """
        Returns the set of all `train_micro_batch_size_per_gpu` (mbs) used in the Trial's lineage.
        """
        mbs_in_lineage = {t.mbs for t in self.lineage_set}
        return mbs_in_lineage

    @property
    def stage(self) -> int:
        return self.ds_config.get("zero_optimization", {}).get("stage", 0)

    @property
    def fp16(self) -> int:
        return self.ds_config.get("fp16", {}).get("enabled") or False

    @property
    def mbs(self) -> int:
        return self.ds_config["train_micro_batch_size_per_gpu"]

    @property
    def create_and_val_ops(self) -> List[searcher.Operation]:
        """
        Returns a list with the Create and ValidateAfter operations needed to initiate and run
        the specified Trial.
        """
        create_op = searcher.Create(
            request_id=self.request_id,
            hparams=self.hparams,
            checkpoint=None,
        )
        validate_after_op = searcher.ValidateAfter(request_id=self.request_id, length=self.length)
        ops_list = [create_op, validate_after_op]

        return ops_list

    @property
    def searcher_metric_val(self) -> Any:
        return self.metric.get(self.searcher_metric_name)

    # TODO: More important properties, like train_batch_size, gas, etc.


class DSATModelProfileInfoTrial(DSATTrial):
    """
    Super class for differentiating the model profiling info run.
    """


class DSATTrialTracker:
    """
    Class for organizing DSATTrial instances and retrieving pertinent info.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        exp_config: Dict[str, Any],
    ) -> None:
        self.exp_config = exp_config
        self.max_trials = args.max_trials
        self.max_concurrent_trials = args.max_concurrent_trials
        self.max_slots = args.max_slots
        self.early_stopping = args.early_stopping
        self.model_dir = args.model_dir
        self.searcher_metric = args.metric
        self.start_profile_step = args.start_profile_step
        self.end_profile_step = args.end_profile_step
        self.zero_stages = set(args.zero_stages)

        # Derived attributes
        self.slots_per_trial = self.exp_config["resources"]["slots_per_trial"]
        self.hparams = self.exp_config["hyperparameters"]

        self.smaller_is_better = _utils.smaller_is_better(self.searcher_metric)

        self.model_profile_info_trial = None
        self.num_trials_since_best_result = 0
        self.successful_stages = set()
        self._all_trials_dict = {}
        self._trial_queue = deque()

        self._mem_per_gpu_per_stage = None
        self._approx_max_mbs_per_stage = None

    def __len__(self) -> int:
        return len(self._all_trials_dict)

    def __getitem__(self, request_id: uuid.UUID) -> DSATTrial:
        return self._all_trials_dict[request_id]

    def __iter__(self) -> Iterator[DSATTrial]:
        return iter(self._all_trials_dict.items())

    def __contains__(self, item: Union[uuid.UUID, DSATTrial]) -> bool:
        if isinstance(item, uuid.UUID):
            return item in self._all_trials_dict
        elif isinstance(item, DSATTrial):
            return item in self._all_trials_dict.values()
        else:
            raise ValueError(
                f"Expected a `uuid.UUID` or `DSATTrial` instance, instead received an object of type {type(item)}"
            )

    def create_trial(
        self,
        hparams: Dict[str, Any],
        search_data: Optional[Any] = None,
        parent_trial: Optional[DSATTrial] = None,
    ) -> DSATTrial:
        """
        Helper function which creates a new `DSATTrial` object of the appropriate length, given the
        config.
        """
        # Create a consistent batch size configuration which obeys the DS constraints.
        self.enforce_consistent_batch_config(hparams)

        # For some reason, DS (0.8.3) exits in the DeepSpeedEngine.step call when
        # DeepSpeedEngine.global_step (initiated at zero) equals end_profile_step + 1,
        # with global_step updated *before* this check happens. So, we need to run for
        # a length of end_profile_step + 1 to trigger the exit. Presumably an off-by-one error
        # on their end.
        trial = DSATTrial(
            hparams=hparams,
            model_dir=self.model_dir,
            slots_per_trial=self.slots_per_trial,
            length=self.end_profile_step + 1,
            parent=parent_trial,
            search_data=search_data,
            searcher_metric_name=self.searcher_metric,
        )
        return trial

    def create_model_profile_info_trial(
        self,
    ) -> DSATModelProfileInfoTrial:
        # Create the special hp dictionary used for the model profile info run.
        model_profile_info_hps = copy.deepcopy(self.hparams)
        model_profile_info_hps[_defaults.OVERWRITE_KEY] = merge_dicts(
            model_profile_info_hps.get(_defaults.OVERWRITE_KEY, {}),
            _defaults.MODEL_INFO_PROFILE_DS_CONFIG,
        )
        self.enforce_consistent_batch_config(model_profile_info_hps)

        model_profile_info_trial = DSATModelProfileInfoTrial(
            hparams=model_profile_info_hps,
            model_dir=self.model_dir,
            slots_per_trial=self.slots_per_trial,
            length=1,  # Only need a single step.
        )
        self.model_profile_info_trial = model_profile_info_trial
        return model_profile_info_trial

    def register_trial(self, trial: DSATTrial) -> None:
        self._all_trials_dict[trial.request_id] = trial
        self._trial_queue.append(trial)

    def pop_next_trial(self) -> DSATTrial:
        """
        Pops the next trial off of the queue and sets its `running` attrribute to `True`.
        """
        next_trial = self._trial_queue.popleft()
        next_trial.running = True
        return next_trial

    def enforce_consistent_batch_config(self, hparams: Dict[str, Any]) -> None:
        """Enforces a consistent batch size configuration by altering `hparams` in-place."""
        # TODO: Talk to Liam about this, because this function adjusts `train_batch_size`, whereas
        # he probably wants this to be the only constant, in order to hold training dynamics fixed.
        # We are optimizing different things.
        ds_config = _utils.get_ds_config_from_hparams(hparams, self.model_dir)
        batch_size_config = _utils.get_batch_config_from_mbs_gas_and_slots(
            ds_config, slots=self.slots_per_trial
        )
        hparams[_defaults.OVERWRITE_KEY] = merge_dicts(
            hparams[_defaults.OVERWRITE_KEY], batch_size_config
        )

    def update_trial_metric(
        self,
        trial: DSATTrial,
        metric: Dict[str, Any],
    ) -> None:
        """
        Updates the Trial Tracker after metrics have been reported, attaching the reported metrics
        to the `DSATTrial` instnace and updating early-stopping bookkeeping.
        """
        trial.metric = metric

        # The model info profiling run's metric will not contain the searcher metric key and should
        # not be counted against the early stopping criteria.
        if not isinstance(trial, DSATModelProfileInfoTrial):
            self.successful_stages.add(trial.stage)
            trial_is_best = self.best_trial == trial
            if trial_is_best:
                self.num_trials_since_best_result = 0
            else:
                self.num_trials_since_best_result += 1
        trial.closed = True
        trial.running = False

    def report_trial_early_exit(self, trial: DSATTrial) -> None:
        trial.error = True
        trial.closed = True
        trial.running = False

    @property
    def gpu_mem(self) -> int:
        """
        Returns the available GPU memory in bytes.
        """
        assert (
            self.model_profile_info_trial is not None
        ), "The model profile info Trial must be run before calling this method."
        return self.model_profile_info_trial.metric["gpu_mem"]

    @property
    def num_params(self) -> int:
        assert (
            self.model_profile_info_trial is not None
        ), "The model profile info Trial must be run before calling this method."
        return self.model_profile_info_trial.metric["num_params"]

    @property
    def trainable_num_params(self) -> int:
        assert (
            self.model_profile_info_trial is not None
        ), "The model profile info Trial must be run before calling this method."
        return self.model_profile_info_trial.metric["trainable_num_params"]

    @property
    def activation_mem_per_gpu(self) -> int:
        assert (
            self.model_profile_info_trial is not None
        ), "The model profile info Trial must be run before calling this method."
        return self.model_profile_info_trial.metric["activation_mem_per_gpu"]

    @property
    def mem_per_gpu_per_stage(self) -> Dict[int, int]:
        """
        Returns the required gpu memory in bytes, per stage, according to whether fp16 training was
        used (other low-precision cases not handled).
        """
        assert (
            self.model_profile_info_trial is not None
        ), "The model profile info Trial must be run before calling this method."
        fp16 = self.model_profile_info_trial.fp16
        if self._mem_per_gpu_per_stage is None:
            params_mem = self.num_params * (2 if fp16 else 4)
            # Gradients must be converted to fp32 to update master weights, so they eventually
            # require the same memory regardless of whether mixed-precision is used.
            gradients_mem = self.trainable_num_params * 4
            # optimizer_mem assumes Adam, following DS. TODO: don't assume this.
            master_params_mem = 4 if fp16 else 0
            momentum_mem = variance_mem = 4
            optimizer_mem = self.trainable_num_params * (
                master_params_mem + momentum_mem + variance_mem
            )

            non_activation_mem_per_gpu_per_stage = {
                0: params_mem + gradients_mem + optimizer_mem,
                1: params_mem + gradients_mem + optimizer_mem // self.slots_per_trial,
                2: params_mem + (gradients_mem + optimizer_mem) // self.slots_per_trial,
                3: (params_mem + gradients_mem + optimizer_mem) // self.slots_per_trial,
            }
            # In DS there is an mp_size int which can be used for model parallelism and also enters
            # the memory computation, but we will not support that feature at the moment.

            mem_per_gpu_per_stage = {
                stage: mem + self.activation_mem_per_gpu
                for stage, mem in non_activation_mem_per_gpu_per_stage.items()
            }
            self._mem_per_gpu_per_stage = mem_per_gpu_per_stage
        return self._mem_per_gpu_per_stage

    @property
    def approx_max_mbs_per_stage(self) -> Dict[int, int]:
        """
        Returns the approximate max train_micro_batch_size_per_gpu (mbs) per stage.

        """
        if self._approx_max_mbs_per_stage is None:
            self._approx_max_mbs_per_stage = {
                stage: max((self.gpu_mem - mem) // self.activation_mem_per_gpu, 1)
                for stage, mem in self.mem_per_gpu_per_stage.items()
            }
        return self._approx_max_mbs_per_stage

    def _best_trial_fn(self, trials: Iterable[DSATTrial]) -> DSATTrial:
        trials_with_searcher_metric = [
            trial
            for trial in trials
            if self.searcher_metric in trial.metric
            and not isinstance(trial, DSATModelProfileInfoTrial)
        ]
        if not trials_with_searcher_metric:
            return None

        min_or_max = min if self.smaller_is_better else max
        best_trial = min_or_max(
            trials_with_searcher_metric, key=lambda trial: trial.metric[self.searcher_metric]
        )
        return best_trial

    @property
    def best_trials_by_stage(self) -> Dict[str, DSATTrial]:
        best_trials_by_stage = {
            stage: self._best_trial_fn(trial for _, trial in self if trial.stage == stage)
            for stage in range(4)
        }
        return best_trials_by_stage

    @property
    def best_trial(self) -> DSATTrial:
        best_trial = self._best_trial_fn(
            trial for trial in self.best_trials_by_stage.values() if trial is not None
        )
        return best_trial

    @property
    def running_trials(self) -> List[DSATTrial]:
        return [trial for _, trial in self if trial.running]

    @property
    def closed_trials(self) -> List[DSATTrial]:
        return [trial for _, trial in self if trial.closed]

    @property
    def num_running_trials(self) -> int:
        return len(self.running_trials)

    @property
    def num_closed_trials(self) -> int:
        return len(self.closed_trials)

    @property
    def num_trials_in_queue(self) -> int:
        return len(self._trial_queue)

    @property
    def max_trials_queued(self) -> bool:
        return self.num_trials_in_queue >= self.max_trials

    @property
    def empty_queue(self) -> bool:
        return not self.num_trials_in_queue

    @property
    def max_trials_are_running_or_closed(self) -> bool:
        return self.num_running_trials + self.num_closed_trials >= self.max_trials

    @property
    def early_stopping_triggered(self) -> bool:
        if self.early_stopping is None:
            return False
        return self.num_trials_since_best_result >= self.early_stopping

    @property
    def should_shutdown(self) -> bool:
        if self.model_profile_info_trial is not None and self.model_profile_info_trial.error:
            logging.info("Shutting down: error in model profile info Trial.")
            return True
        elif self.early_stopping_triggered:
            logging.info("Shutting down: early stopping criteria met.")
            return True
        elif self.num_closed_trials >= self.max_trials:
            logging.info("Shutting down: all Trials completed.")
            return True
        else:
            return False

    @property
    def should_be_failure(self) -> bool:
        model_profile_info_trial_failed = (
            self.model_profile_info_trial is not None and self.model_profile_info_trial.error
        )
        every_autotuning_trial_failed = all(
            trial.error
            for _, trial in self
            if trial.closed and not isinstance(trial, DSATModelProfileInfoTrial)
        )
        return model_profile_info_trial_failed or every_autotuning_trial_failed

    @property
    def can_run_more_trials(self) -> int:
        if self.empty_queue:
            return False
        if self.max_trials_are_running_or_closed:
            return False
        if self.num_running_trials >= self.max_concurrent_trials:
            return False
        if self.max_slots is not None:
            occupied_slots = self.num_running_trials * self.slots_per_trial
            remaining_slots = self.max_slots - occupied_slots
            trials_available_with_remaining_slots = remaining_slots // self.slots_per_trial
            return trials_available_with_remaining_slots > 0
        return True


class BaseDSATSearchMethod(searcher.SearchMethod):
    """
    Base class for all DS AT searchers. Written so that only the `get_new_searcher_ops_list` method
    needs to be written overwritten when subclassing (at a minimum).
    """

    def __init__(self, args: argparse.Namespace, exp_config: Dict[str, Any]) -> None:
        # Storing args so that additional args can be inherited by child classes
        self.args = args
        self.exp_config = exp_config
        self.trial_tracker = DSATTrialTracker(args=args, exp_config=exp_config)
        self.rng = np.random.default_rng(seed=args.random_seed)
        random.seed(args.random_seed)

        self._tracker_ckpt_path = "trial_tracker.pkl"
        self._py_rand_ckpt_path = "py_random_state.pkl"
        self._np_rand_ckpt_path = "np_rng.pkl"

    @abstractmethod
    def get_trials_after_validation_completed(
        self,
        searcher_state: searcher.SearcherState,
        last_trial: DSATTrial,
        metric: Union[float, Dict[str, Any]],
    ) -> Iterable[DSATTrial]:
        """
        To be defined in all subclasses.

        Generates a list of new operations to run based on the results of the last successful trial.
        """
        pass

    @abstractmethod
    def get_trials_after_early_exit(
        self,
        searcher_state: searcher.SearcherState,
        last_trial: DSATTrial,
        exited_reason: searcher.ExitedReason,
    ) -> Iterable[DSATTrial]:
        """
        To be defined in all subclasses.

        Generates a list of new operations to run after the last trial exited early.
        """
        pass

    def initial_operations(
        self, searcher_state: searcher.SearcherState
    ) -> List[searcher.Operation]:
        """
        Submits the model info profiling run in order to collect model and resources info to
        inform the search.
        """
        # TODO: Remove print tests.
        logging.info("Initial operations")
        self._searcher_state_tests(searcher_state, "inital ops")

        model_profile_info_trial = self.trial_tracker.create_model_profile_info_trial()
        self.trial_tracker.register_trial(model_profile_info_trial)
        self.trial_tracker.pop_next_trial()  # Needed for bookkeeping.
        ops = model_profile_info_trial.create_and_val_ops
        return ops

    def on_trial_created(
        self, searcher_state: searcher.SearcherState, request_id: uuid.UUID
    ) -> List[searcher.Operation]:
        # TODO: Remove print tests.
        logging.info("on trial created")
        self._searcher_state_tests(searcher_state, "trial created")

        return []

    def on_validation_completed(
        self,
        searcher_state: searcher.SearcherState,
        request_id: uuid.UUID,
        metric: Union[float, Dict[str, Any]],
        train_length: int,
    ) -> List[searcher.Operation]:
        last_trial = self.trial_tracker[request_id]
        self.trial_tracker.update_trial_metric(trial=last_trial, metric=metric)

        # TODO: remove some of these info logs. Some are just for testing.
        if isinstance(last_trial, DSATModelProfileInfoTrial):
            logging.info(
                f"Approx. max mbs per stage: {self.trial_tracker.approx_max_mbs_per_stage}"
            )
            logging.info(
                f"Approx. GPU memory per stage: {self.trial_tracker.mem_per_gpu_per_stage}"
            )
            logging.info(f"Total GPU memory: {self.trial_tracker.gpu_mem}")

        if not self.trial_tracker.max_trials_queued and not self.trial_tracker.should_shutdown:
            new_trials = self.get_trials_after_validation_completed(
                searcher_state=searcher_state,
                last_trial=last_trial,
                metric=metric,
            )
            for trial in new_trials:
                self.trial_tracker.register_trial(trial)

        # TODO: Remove print tests.
        logging.info(f"Calling on_validation_completed for {request_id}")
        self._searcher_state_tests(searcher_state, "val completed")

        # All DS AT Trials should be closed after validation.
        return [searcher.Close(request_id)]

    def on_trial_exited_early(
        self,
        searcher_state: searcher.SearcherState,
        request_id: uuid.UUID,
        exited_reason: searcher.ExitedReason,
    ) -> List[searcher.Operation]:
        last_trial = self.trial_tracker[request_id]
        self.trial_tracker.report_trial_early_exit(last_trial)

        # TODO: Remove print tests.
        logging.info(f"Calling on_trial_exited_early for {request_id}")
        self._searcher_state_tests(searcher_state, "exited early")

        new_ops_list = []
        if exited_reason != searcher.ExitedReason.ERRORED:
            # In case of INVALID_HP or USER_CANCELED, shut down the searcher.
            logging.info(
                f"Shutting down: unexpected early exit due to {exited_reason}"
                f"\nLast trial: {last_trial}, request_id: {request_id}"
            )
            new_ops_list.append(searcher.Shutdown(failure=self.trial_tracker.should_be_failure))
        if not self.trial_tracker.max_trials_queued and not self.trial_tracker.should_shutdown:
            # ERRORED Trials generally corresponds to OOMs, after which we may want to submit
            # follow-on Trials.
            new_trials = self.get_trials_after_early_exit(
                searcher_state=searcher_state,
                last_trial=last_trial,
                exited_reason=exited_reason,
            )
            for trial in new_trials:
                self.trial_tracker.register_trial(trial)

        return new_ops_list

    def on_trial_closed(
        self, searcher_state: searcher.SearcherState, request_id: uuid.UUID
    ) -> List[searcher.Operation]:
        last_trial = self.trial_tracker[request_id]

        # TODO: Remove print tests.
        logging.info(f"Calling on_trial_closed for {request_id}")
        logging.info(f"metrics for closed trial {last_trial.metric}")

        new_ops_list = []
        if self.trial_tracker.should_shutdown:
            if self.trial_tracker.best_trial is not None and self.args.run_full_experiment:
                submitted_config = _utils.get_dict_from_yaml_or_json_path(self.args.config_path)
                optimal_config = merge_dicts(
                    submitted_config, {"hyperparameters": self.trial_tracker.best_trial.hparams}
                )
                # Delete the keys which enforce autotuning code paths
                del optimal_config["hyperparameters"][_defaults.OVERWRITE_KEY]["autotuning"]
                del optimal_config["hyperparameters"][_defaults.USE_DSAT_MODE_KEY]
                # TODO: add searcher exp_id to the config so the user knows where this came from
                # and also some "optimal config" label somewhere.
                create_experiment(optimal_config, self.args.model_dir, self.args.include)

            new_ops_list.append(searcher.Shutdown(failure=self.trial_tracker.should_be_failure))
        else:
            while self.trial_tracker.can_run_more_trials:
                next_trial = self.trial_tracker.pop_next_trial()
                new_ops_list.extend(next_trial.create_and_val_ops)

        self._searcher_state_tests(searcher_state, "trial closed")

        return new_ops_list

    def progress(self, searcher_state: searcher.SearcherState) -> float:
        # TODO: Remove print tests.
        logging.info("progress")
        self._searcher_state_tests(searcher_state, "progress")

        progress = len(searcher_state.trials_closed) / len(searcher_state.trials_created)
        return progress

    def save_method_state(self, path: pathlib.Path) -> None:
        with path.joinpath(self._tracker_ckpt_path).open("wb") as f:
            pickle.dump(self.trial_tracker, f)
        with path.joinpath(self._py_rand_ckpt_path).open("wb") as f:
            pickle.dump(random.getstate(), f)
        with path.joinpath(self._np_rand_ckpt_path).open("wb") as f:
            pickle.dump(self.rng, f)
        if self.trial_tracker.best_trial is not None:
            with path.joinpath("best_ds_config.json").open("w") as f:
                json.dump(self.trial_tracker.best_trial.ds_config, f)

    def load_method_state(self, path: pathlib.Path) -> None:
        logging.info("Restoring searcher state from checkpoint.")
        with path.joinpath(self._tracker_ckpt_path).open("rb") as f:
            self.trial_tracker = pickle.load(f)
        with path.joinpath(self._py_rand_ckpt_path).open("rb") as f:
            py_random_state = pickle.load(f)
            random.setstate(py_random_state)
        with path.joinpath(self._np_rand_ckpt_path).open("rb") as f:
            self.rng = pickle.load(f)

    def _searcher_state_tests(
        self,
        searcher_state: searcher.SearcherState,
        text: str,
    ) -> None:
        # for testing, delete later

        running_trials = searcher_state.trials_created - searcher_state.trials_closed
        num_running_trials = len(running_trials)
        trials_created = len(searcher_state.trials_created)
        trials_created_in_tracker = len(self.trial_tracker)
        total_trials_remaining = self.trial_tracker.max_trials - trials_created

        concurrent_trials_available = self.trial_tracker.max_concurrent_trials - num_running_trials
        total_slots = self.trial_tracker.slots_per_trial * num_running_trials
        logging.info(f"running trials (SearcherState, {text}): {num_running_trials}")
        logging.info(f"trials created (SearcherState, {text}): {trials_created}")
        logging.info(
            f"trials created in tracker (SearcherState, {text}): {trials_created_in_tracker}"
        )
        logging.info(f"trials closed (SearcherState, {text}): {len(searcher_state.trials_closed)}")
        logging.info(f"trials remaining (SearcherState, {text}): {total_trials_remaining}")
        logging.info(
            f"Concurrent trials remaining (SearcherState, {text}): {concurrent_trials_available}"
        )
        logging.info(f"total slots (SearcherState, {text}): {total_slots}")

        assert (
            num_running_trials <= self.trial_tracker.max_concurrent_trials
        ), f"running trials {num_running_trials}, limit {self.trial_tracker.max_concurrent_trials}, {running_trials}"
        if self.trial_tracker.max_slots is not None:
            assert (
                total_slots <= self.trial_tracker.max_slots
            ), f"total slots {total_slots}, limit {self.trial_tracker.max_slots}, {running_trials}"
        assert (
            len(searcher_state.trials_created) <= self.trial_tracker.max_trials
        ), f"total trials {trials_created}, limit {self.trial_tracker.max_trials}, {running_trials}"


class RandomDSATSearchMethod(BaseDSATSearchMethod):
    """
    Semi-random search through parameters space. Attaches search_data of the form
    {"lo": lo,  "hi": hi} which defines the inclusive bounds on the train_micro_batch_size_per_gpu
    that can be selected for the trial.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Save/restore rng state in checkpoints.
        self.trials_per_random_config = self.args.trials_per_random_config

    def get_trials_after_validation_completed(
        self,
        searcher_state: searcher.SearcherState,
        last_trial: DSATTrial,
        metric: Optional[Union[float, Dict[str, Any]]] = None,
    ) -> List[DSATTrial]:
        new_trials = []
        if isinstance(last_trial, DSATModelProfileInfoTrial):
            new_trials = self.get_trial_list_after_model_profile_info_run()
        else:
            new_trials = self.get_trial_list_after_successful_run(last_trial)
        return new_trials

    def get_trials_after_early_exit(
        self,
        searcher_state: searcher.SearcherState,
        last_trial: DSATTrial,
        exited_reason: searcher.ExitedReason,
    ) -> List[DSATTrial]:
        # TODO: delete print test
        logging.info("Calling get_trials_after_early_exit")

        if self.should_stop_lineage(last_trial):
            new_trials = [self.get_random_trial()]
        else:
            # Lower the ceiling after the failure.
            new_search_data = {"lo": last_trial.search_data["lo"], "hi": last_trial.mbs - 1}

            mbs = self.get_random_mbs_from_search_data(new_search_data)
            new_hparams = copy.deepcopy(last_trial.hparams)
            new_hparams[_defaults.OVERWRITE_KEY]["train_micro_batch_size_per_gpu"] = mbs

            trial = self.trial_tracker.create_trial(
                hparams=new_hparams,
                search_data=new_search_data,
                parent_trial=last_trial,
            )
            new_trials = [trial]
        return new_trials

    def get_trial_list_after_model_profile_info_run(self) -> List[DSATTrial]:
        new_trials = []
        num_remaining_trials = self.trial_tracker.max_trials - 1
        num_initial_lineages = num_remaining_trials // self.trials_per_random_config
        for _ in range(num_initial_lineages):
            trial = self.get_random_trial()
            new_trials.append(trial)
        return new_trials

    def get_trial_list_after_successful_run(
        self,
        last_trial: DSATTrial,
    ) -> List[DSATTrial]:
        # TODO: remove below print tests.
        logging.info("**************** BSZ History ****************")
        bsz_history = []
        print_trial = last_trial
        while print_trial is not None:
            bsz = print_trial.ds_config["train_micro_batch_size_per_gpu"]
            bsz_history.append(bsz)
            print_trial = print_trial.parent
        logging.info(f"History (to-be-submitted last): {str(list(reversed(bsz_history)))}")
        logging.info("**************** BSZ History End ****************")

        # TODO: verify we are always quitting when no more non-trivial trials are possible.
        if self.should_stop_lineage(last_trial=last_trial):
            return [self.get_random_trial()]

        # Let the best trial for this stage inform the next one, if it exists.
        best_trial_for_stage = self.trial_tracker.best_trials_by_stage[last_trial.stage]
        if best_trial_for_stage is not None:
            new_search_data = copy.deepcopy(best_trial_for_stage.search_data)
            # Update the floor to one greater than the mbs used.
            new_search_data["lo"] = best_trial_for_stage.mbs + 1
        else:
            # Otherwise, start from the data from the last trial.
            new_search_data = copy.deepcopy(last_trial.search_data)
            new_search_data["lo"] = last_trial.mbs + 1
            # The initial `hi` ceiling is just a soft guess, so double it, if the ceiling was
            # successfully run.
            if last_trial.mbs == last_trial.search_data["hi"]:
                new_search_data["hi"] *= 2

        mbs = self.get_random_mbs_from_search_data(new_search_data)
        # TODO: Check we haven't run this experiment before.

        new_hparams = copy.deepcopy(last_trial.hparams)
        new_hparams[_defaults.OVERWRITE_KEY]["train_micro_batch_size_per_gpu"] = mbs
        trial = self.trial_tracker.create_trial(
            hparams=new_hparams,
            search_data=new_search_data,
            parent_trial=last_trial,
        )
        return [trial]

    def should_stop_lineage(self, last_trial: DSATTrial) -> bool:
        # Generic stopping conditions first
        if last_trial.num_trials_in_lineage >= self.trials_per_random_config:
            return True

        # DS domain knowledge: if stages 1 or 2 run successfully, there is no need to use stage 3.
        stage_one_or_two_successful = {1, 2} & self.trial_tracker.successful_stages
        should_stop_this_stage_3_trial = last_trial.stage == 3 and stage_one_or_two_successful
        if should_stop_this_stage_3_trial:
            return True

        if last_trial.error:
            failed_on_minimal_batch = last_trial.mbs == last_trial.search_data["lo"]
            if failed_on_minimal_batch:
                return True

            successful_configs_at_same_mbs_and_stage = any(
                trial.mbs >= last_trial.mbs
                for request_id, trial in self.trial_tracker
                if request_id != last_trial.request_id
                and trial.stage == last_trial.stage
                and trial.searcher_metric_val
            )
            if successful_configs_at_same_mbs_and_stage:
                return True

        elif not last_trial.error:

            def better(trial1, trial2):
                if self.trial_tracker.smaller_is_better:
                    return trial1.searcher_metric_val < trial2.searcher_metric_val
                else:
                    return trial1.searcher_metric_val > trial2.searcher_metric_val

            better_configs_at_same_mbs_and_stage = any(
                better(trial, last_trial)
                for request_id, trial in self.trial_tracker
                if request_id != last_trial.request_id
                and trial.stage == last_trial.stage
                and trial.searcher_metric_val
            )
            if better_configs_at_same_mbs_and_stage:
                return True

        return False

    def get_random_mbs_from_search_data(self, search_data: Dict[str, int]) -> int:
        mbs = search_data["lo"] + self.rng.binomial(search_data["hi"] - search_data["lo"], 0.5)
        assert search_data["lo"] <= mbs <= search_data["hi"]  # TODO: remove
        return mbs

    def get_random_hparams_and_search_data(
        self, zero_stage
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        zero_optim_config = _utils.get_random_zero_optim_config(zero_stage)
        new_hparams = copy.deepcopy(self.trial_tracker.hparams)
        new_hparams[_defaults.OVERWRITE_KEY] = merge_dicts(
            new_hparams.get(_defaults.OVERWRITE_KEY, {}),
            {"zero_optimization": zero_optim_config},
        )

        # If a best trial has been established for the given stage, use its search data bounds to
        # choose a better starting point.
        best_trial_for_stage = self.trial_tracker.best_trials_by_stage[zero_stage]
        if best_trial_for_stage is not None:
            new_search_data = copy.deepcopy(best_trial_for_stage.search_data)
            # Update the floor to one greater than the mbs used.
            new_search_data["lo"] = self.trial_tracker.best_trial.mbs + 1
            # If the new floor is higher than the ceiling, double the ceiling
            while new_search_data["lo"] >= new_search_data["hi"]:
                new_search_data["hi"] *= 2
        # Otherwise choose the corrsponding search data based on approximate computations
        else:
            random_zero_stage_max_mbs = self.trial_tracker.approx_max_mbs_per_stage[zero_stage]
            new_search_data = {
                "lo": 1,
                "hi": random_zero_stage_max_mbs,
            }

        # Randomly choose the actual batch size.
        mbs = self.get_random_mbs_from_search_data(new_search_data)
        new_hparams[_defaults.OVERWRITE_KEY]["train_micro_batch_size_per_gpu"] = mbs
        return (new_hparams, new_search_data)

    def get_random_trial(self) -> DSATTrial:
        # Choose the stage randomly from user provided stages, after some performance filtering.
        # If stage one or two was successful, don't continue with stage 3.
        stage_one_and_two = {1, 2}
        successful_one_or_two_stages = stage_one_and_two & self.trial_tracker.successful_stages
        filtered_zero_stages = successful_one_or_two_stages & self.trial_tracker.zero_stages
        if filtered_zero_stages:
            zero_stage = random.choice(list(filtered_zero_stages))
        else:
            zero_stage = random.choice(list(self.trial_tracker.zero_stages))

        hparams, search_data = self.get_random_hparams_and_search_data(zero_stage)
        random_trial = self.trial_tracker.create_trial(hparams=hparams, search_data=search_data)
        return random_trial


class SimpleDSATSearchMethod(BaseDSATSearchMethod):
    """
    Dumb searcher which just submits Trials with linearly increasing batch sizes, from 2 up to
    max_trials
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_trials_after_validation_completed(
        self,
        searcher_state: searcher.SearcherState,
        last_trial: DSATTrial,
        metric: Optional[Union[float, Dict[str, Any]]] = None,
    ) -> List[DSATTrial]:
        new_trials = []
        if isinstance(last_trial, DSATModelProfileInfoTrial):
            # Delete special DS keys which force a model profiling info run.
            hparams_without_profile_info_keys = last_trial.hparams
            del hparams_without_profile_info_keys[_defaults.OVERWRITE_KEY]["autotuning"][
                "model_info"
            ]
            del hparams_without_profile_info_keys[_defaults.OVERWRITE_KEY]["autotuning"][
                "model_info_path"
            ]
            for tmbs in range(2, self.trial_tracker.max_trials + 1):
                hparams = copy.deepcopy(hparams_without_profile_info_keys)
                hparams[_defaults.OVERWRITE_KEY]["train_micro_batch_size_per_gpu"] = tmbs
                # Choose a random zero stage:
                hparams[_defaults.OVERWRITE_KEY]["zero_optimization"] = {
                    "stage": random.choice(list(self.args.zero_stages))
                }
                trial = self.trial_tracker.create_trial(
                    hparams=hparams,
                    search_data=None,
                    parent_trial=None,
                )
                new_trials.append(trial)
        return new_trials

    def get_trials_after_early_exit(
        self,
        searcher_state: searcher.SearcherState,
        last_trial: DSATTrial,
        exited_reason: searcher.ExitedReason,
    ) -> List[DSATTrial]:
        return []
