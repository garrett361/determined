import argparse
import copy
import logging
import uuid
from typing import Any, Dict, List

import determined as det
from determined import searcher
from dsat import constants, utils


class DSATSearchMethod(searcher.SearchMethod):
    def __init__(self, profiling_results_dict: Dict[str, Any], original_config_dict) -> None:
        self.profiling_results_dict = profiling_results_dict
        self.original_config_dict = original_config_dict
        self.running_trials = 0

    def _get_list_of_hparams(self) -> List[Dict[str, Any]]:
        """Generates a list of all hp dict combos which will be tested out."""
        return 2 * [self.original_config_dict]

    ############################################################################
    # Invoked only once, when starting a new experiment. Creates initial list
    # of operations.
    # In this example, we create and submit operations for first N trials, such that:
    #   1) each trial is assigned a unique request_id and every operation
    #      contains request_id of a trial it refers to;
    #   2) each trial is initialized with two operations:
    #      -> "Create" operation that takes in trial's request_id and hyperparameters;
    #         in this example hyperparameters are generated by user-defined method
    #         search_space(),
    #      -> "ValidateAfter" operation that takes in trial's request_id and number of
    #         units (batches or epochs) that the model is trained for before validation;
    #         units selection is made in the custom_config.yaml.
    #
    # Note: the order in which trials are created is not guaranteed.
    def initial_operations(self, _: searcher.SearcherState) -> List[searcher.Operation]:
        operations = []
        for hp_dict in self._get_list_of_hparams():
            create = searcher.Create(
                request_id=uuid.uuid4(),
                hparams=hp_dict,
                checkpoint=None,
            )
            run = searcher.ValidateAfter(
                request_id=create.request_id, length=constants.MODEL_INFO_MAX_LENGTH
            )
            operations.append(create)
            operations.append(run)

        return operations

    def on_trial_created(
        self, _: searcher.SearcherState, request_id: uuid.UUID
    ) -> List[searcher.Operation]:
        self.running_trials += 1
        print(f"Creating trial {request_id}, {self.running_trials} remaining")
        return []

    def on_validation_completed(
        self, _: searcher.SearcherState, request_id: uuid.UUID, metric: float, train_length: int
    ) -> List[searcher.Operation]:
        print(f"Completed trial {request_id}")
        return [searcher.Close(request_id=request_id)]

    def on_trial_closed(
        self, _: searcher.SearcherState, request_id: uuid.UUID
    ) -> List[searcher.Operation]:
        self.running_trials -= 1
        print(f"Closing trial {request_id}, {self.running_trials} remaining")
        if not self.running_trials:
            return [searcher.Shutdown()]
        return []

    def on_trial_exited_early(
        self,
        _: searcher.SearcherState,
        request_id: uuid.UUID,
        exited_reason: searcher.ExitedReason,
    ) -> List[searcher.Operation]:
        return []

    def progress(self, _: searcher.SearcherState) -> float:
        return 0


def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default="")

    parsed_args = parser.parse_args()

    return parsed_args


def main(core_context: det.core.Context) -> None:
    info = det.get_cluster_info()
    model_info_profiling_results_dict = info.trial.hparams["results"]

    args = get_parsed_args()

    config_dict = utils.get_config_dict_from_yaml_path(args.config_path)
    # Instantiate your implementation of SearchMethod
    search_method = DSATSearchMethod(model_info_profiling_results_dict, config_dict)

    # Instantiate RemoteSearchRunner
    search_runner = searcher.RemoteSearchRunner(search_method, context=core_context)

    copy_config_dict = copy.deepcopy(config_dict)
    copy_config_dict["name"] += " (autotuning trial results)"
    copy_config_dict["resources"] = {"slots_per_trial": 0}
    copy_config_dict["entrypoint"] = "python3 -m dsat.dsat_experiment_wrapper"
    print("dict used in search_runner.run", copy_config_dict)
    search_runner.run(copy_config_dict, model_dir=".")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    with det.core.init() as core_context:
        main(core_context)
