import argparse
import collections
import copy
import json
import logging
import pathlib
import random
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import filelock
import torch
from ruamel import yaml

import determined as det
from determined.pytorch.dsat import _defaults
from determined.util import merge_dicts

CURR_DIR = pathlib.Path(".")


def get_base_parser() -> argparse.ArgumentParser:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("config_path", help="experiment config file (.yaml)")
    base_parser.add_argument("model_dir", help="file or directory containing model definition")
    base_parser.add_argument(
        "-i",
        "--include",
        type=str,
        nargs="+",
        help="additional files to copy into the task container",
    )

    base_parser.add_argument(
        "-mt",
        "--max-trials",
        type=int,
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["max-trials"],
        help="Maximum number of Trials to run, including the model profile info Trial",
    )
    base_parser.add_argument(
        "-ms",
        "--max-slots",
        type=int,
        help="Maximum number of slots to use concurrently across Trials",
    )
    base_parser.add_argument(
        "-mct",
        "--max-concurrent-trials",
        type=int,
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["max-concurrent-trials"],
        help="Maximum number of Trials to run concurrently",
    )
    base_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["metric"],
        choices=_defaults.SMALLER_IS_BETTER_METRICS + _defaults.LARGER_IS_BETTER_METRICS,
    )
    base_parser.add_argument("--run-full-experiment", action="store_true")
    base_parser.add_argument(
        "-z",
        "--zero-stages",
        type=int,
        nargs="+",
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["zero-stages"],
        choices=list(range(4)),
        help="Space-separated list of zero stages to search over",
    )
    base_parser.add_argument(
        "--start_profile-step",
        type=int,
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["start-profile-step"],
        help="Step on which to start profiling",
    )
    base_parser.add_argument(
        "--end-profile-step",
        type=int,
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["end-profile-step"],
        help="Step on which to stop profiling",
    )
    base_parser.add_argument(
        "-r",
        "--random_seed",
        type=int,
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["random-seed"],
    )
    base_parser.add_argument(
        "--search-runner-config",
        type=str,
        help="Path to an alternative search runner configuration file. For advanced use cases",
    )
    base_parser.add_argument(
        "--max-search-runner-restarts", type=int, default=5, help="Maximum search runner restarts"
    )

    return base_parser


def get_full_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="Determined AI DeepSpeed Autotune")
    subparsers = parser.add_subparsers(required=True, dest="search_method")
    base_parser = get_base_parser()

    subparsers.add_parser("_test", parents=[base_parser])

    random_subparser = subparsers.add_parser("random", parents=[base_parser])
    random_subparser.add_argument(
        "--trials-per-random-config",
        type=int,
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["trials-per-random-config"],
        help="Maximum number of Trials to run per random config",
    )
    random_subparser.add_argument(
        "--early-stopping",
        type=int,
        help="Terminates the search if a new best config not found in last `early-stopping` Trials",
    )

    binary_subparser = subparsers.add_parser("binary", parents=[base_parser])
    search_range_factor_help = (
        "Expands the initial search range by a factor of `search-range-factor`"
    )
    binary_subparser.add_argument(
        "--search-range-factor",
        type=float,
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["search-range-factor"],
        help=search_range_factor_help,
    )

    asha_subparser = subparsers.add_parser("asha", parents=[base_parser])
    asha_subparser.add_argument(
        "--max-rungs",
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["max-rungs"],
        help="Maximum rungs to use in the ASHA algorithm",
    )
    asha_subparser.add_argument(
        "--min-binary-search-trials",
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["min-binary-search-trials"],
        help="Minimum number of binary search Trials to run per random configuration",
    )
    asha_subparser.add_argument(
        "--asha-early-stopping",
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["asha-early-stopping"],
        help="ASHA early stopping parameter (`s` in arxiv:1810.05934)",
    )
    asha_subparser.add_argument(
        "--divisor",
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["divisor"],
        help="ASHA divisor parameter (`eta` in arxiv:1810.05934)",
    )
    asha_subparser.add_argument(
        "--search-range-factor",
        type=float,
        default=_defaults.AUTOTUNING_ARG_DEFAULTS["search-range-factor"],
        help=search_range_factor_help,
    )

    return parser


def smaller_is_better(metric: str) -> bool:
    if metric in _defaults.SMALLER_IS_BETTER_METRICS:
        return True
    elif metric in _defaults.LARGER_IS_BETTER_METRICS:
        return False
    else:
        valid_metrics = _defaults.SMALLER_IS_BETTER_METRICS + _defaults.LARGER_IS_BETTER_METRICS
        raise ValueError(f"metric must be one of {valid_metrics}, not {metric}")


def get_split_entrypoint(submitted_entrypoint: Union[List[str], str]) -> List[str]:
    # The entrypoint may be a string or list of strings. Strip all white space from each entry and
    # convert to a list, in either case.
    if isinstance(submitted_entrypoint, str):
        split_entrypoint = submitted_entrypoint.split(" ")
    elif isinstance(submitted_entrypoint, list):
        # Join and re-split to remove any possile white space.
        # submitted_entrypoint: List[str]
        str_entrypoint: str = " ".join(submitted_entrypoint)
        split_entrypoint = str_entrypoint.split(" ")
    else:
        raise ValueError(
            f"Expected a string or list for an entrypoint, but received "
            f"{type(submitted_entrypoint)}"
        )
    return [s.strip() for s in split_entrypoint if s.strip()]


def get_search_runner_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.search_runner_config is not None:
        submitted_search_runner_config = get_dict_from_yaml_or_json_path(args.search_runner_config)
        return submitted_search_runner_config

    submitted_exp_config_dict = get_dict_from_yaml_or_json_path(args.config_path)
    assert "deepspeed_config" in submitted_exp_config_dict["hyperparameters"], (
        "DS AT requires a `hyperparameters.deepspeed_config` key which points "
        "to the deepspeed config json file"
    )

    # Also sanity check that if a --deepspeed_config (or in the case of HF
    # --deepspeed) arg is passed in, both configs match. Probably some gotchas here because
    # --deepspeed is also a boolean arg for vanilla deepspeed.
    possible_config_flags = ("--deepspeed", "--deepspeed_config")

    submitted_entrypoint: Union[List[str], str] = submitted_exp_config_dict["entrypoint"]
    split_entrypoint = get_split_entrypoint(submitted_entrypoint)

    for idx in range(len(split_entrypoint) - 1):
        curr_arg, next_arg = split_entrypoint[idx : idx + 2]
        next_arg_is_not_a_flag = next_arg != "-"
        if curr_arg in possible_config_flags and next_arg_is_not_a_flag:
            entrypoint_deepspeed_config = next_arg
            hp_deepspeed_config = submitted_exp_config_dict["hyperparameters"]["deepspeed_config"]
            if entrypoint_deepspeed_config != hp_deepspeed_config:
                raise ValueError(
                    f"The deepspeed config path in the `hyperparameters` section, "
                    f"{hp_deepspeed_config}, does not match the path in the entrypoint, "
                    f"{entrypoint_deepspeed_config}."
                )

    default_search_runner_config = _defaults.DEFAULT_SEARCH_RUNNER_CONFIG
    if args.max_search_runner_restarts is not None:
        default_search_runner_config["max_restarts"] = args.max_search_runner_restarts
    # Merge with the submitted experiment config so that the search runner shares the project,
    # workspace, etc.
    search_runner_config = merge_dicts(submitted_exp_config_dict, default_search_runner_config)
    search_runner_config["name"] = f"(DSAT) {search_runner_config['name']}"
    search_runner_config["hyperparameters"] = vars(args)

    return search_runner_config


def get_dict_from_yaml_or_json_path(
    path: str, convert_json_keys_to_int: bool = True
) -> Dict[Any, Any]:
    """
    Load a json or yaml file as a dict. Optionally convert all json dict keys to
    ints, where possible.
    """
    p = pathlib.Path(path)
    if p.suffix == ".json":
        try:
            with open(p, "r") as f:
                json_dict: Dict[Any, Any] = json.load(f)
            if convert_json_keys_to_int:

                def try_str_to_int(s: str) -> Union[str, int]:
                    try:
                        return int(s)
                    except ValueError:
                        return s

                json_dict = {try_str_to_int(k): v for k, v in json_dict.items()}
            return json_dict
        except Exception as e:
            logging.info(f"Exception {e} raised when loading {path} with json. Attempting yaml.")
            return {}
    else:
        with open(p, "r") as f:
            yaml_dict: Dict[Any, Any] = yaml.YAML(typ="safe").load(f)
        return yaml_dict


@contextmanager
def dsat_reporting_context(
    core_context: det.core._context.Context,
    op: det.core._searcher.SearcherOperation,
    steps_completed: Optional[int] = None,
) -> Generator[None, None, None]:
    """
    Context manager required for using Determined AI DeepSpeed Autotune with Core API.

    The `forward` and `step` methods of the DeepSpeed model engine must be called inside of this
    context manager.

    Args:
        core_context: a `Context` instance created with `determined.core.init`
        op: the first `SearcherOperation` instance generated by `core_context.searcher.operations`

    """
    if steps_completed is None:
        steps_completed = op.length
    try:
        yield
    except SystemExit as se:
        model_profiling_path = pathlib.Path(_defaults.MODEL_INFO_PROFILING_PATH)
        autotuning_results_path = pathlib.Path(_defaults.AUTOTUNING_RESULTS_PATH)
        possible_paths = [model_profiling_path, autotuning_results_path]
        existing_paths = [p for p in possible_paths if p.exists()]
        # Exactly one of these files should be generated for each properly exited DS AT Trial.
        if len(existing_paths) == 1:
            path = existing_paths[0]
            add_gpu_info = path == model_profiling_path
            report_json_results(
                core_context=core_context,
                op=op,
                steps_completed=steps_completed,
                add_gpu_info=add_gpu_info,
                path=path,
            )
        raise se


def report_json_results(
    core_context: det.core._context.Context,
    op: det.core._searcher.SearcherOperation,
    steps_completed: int,
    add_gpu_info: bool,
    path: Union[str, pathlib.Path],
) -> None:
    is_chief = core_context.distributed.rank == 0
    if is_chief:
        with open(path, "r") as f:
            results_dict = json.load(f)
        if add_gpu_info:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            results_dict["gpu_mem"] = gpu_mem
        core_context.train.report_validation_metrics(
            steps_completed=steps_completed, metrics=results_dict
        )
        op.report_completed(results_dict)
    # Ensure the operations generator is empty to complete sanity checks.
    try:
        next(core_context.searcher.operations())
    except StopIteration:
        pass
    else:
        raise AssertionError("Unexpected additional operations found!")


def get_zero_stage_search_space(
    zero_stage: int,
) -> Dict[str, List[Union[bool, float]]]:
    default_settings: Dict[
        int, Dict[str, List[Union[bool, float]]]
    ] = _defaults.DEFAULT_ZERO_SEARCH_SPACE
    assert (
        zero_stage in default_settings
    ), f"Invalid zero_stage, must be one of {list(default_settings)}"
    search_space = default_settings[1]
    for stage in range(2, zero_stage + 1):
        search_space = merge_dicts(search_space, default_settings[stage])
    return search_space


def get_random_zero_optim_config(zero_stage: int) -> Dict[str, Union[bool, float]]:
    search_space = get_zero_stage_search_space(zero_stage)
    zero_optim_dict = {k: random.choice(v) for k, v in search_space.items()}
    zero_optim_dict["stage"] = zero_stage
    return zero_optim_dict


def get_batch_config_from_mbs_gas_and_slots(
    ds_config: Dict[str, Any], slots: int
) -> Dict[str, int]:
    """
    Returns a consistent batch size configuration by adjusting `train_batch_size` according to the
    number of `slots`, `train_micro_batch_size_per_gpu`, and `gradient_accumulation_steps`  (or its
    default value, if not specified).
    """
    mbs = ds_config["train_micro_batch_size_per_gpu"]
    gas = ds_config.get("gradient_accumulation_steps", _defaults.GAS_DEFAULT)
    if gas == "auto":
        # Needed for HuggingFace.
        gas = 1
    tbs = mbs * gas * slots
    return {
        "train_batch_size": tbs,
        "train_micro_batch_size_per_gpu": mbs,
        "gradient_accumulation_steps": gas,
    }


def dict_raise_error_on_duplicate_keys(ordered_pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Reject duplicate keys from the ordered_pairs"""
    d = dict(ordered_pairs)
    if len(d) != len(ordered_pairs):
        counter = collections.Counter([pair[0] for pair in ordered_pairs])
        keys = [key for key, value in counter.items() if value > 1]
        raise ValueError("Duplicate keys in DeepSpeed config: {}".format(keys))
    return d


def normalize_base_ds_config(
    base_ds_config: Union[str, Dict[str, Any]], model_dir: pathlib.Path = CURR_DIR
) -> Dict[str, Any]:
    if isinstance(base_ds_config, str):
        full_path = model_dir.joinpath(pathlib.Path(base_ds_config))
        with open(full_path, "r") as f:
            ret_ds_config: Dict[str, Any] = json.load(
                f,
                object_pairs_hook=dict_raise_error_on_duplicate_keys,
            )
        return ret_ds_config
    else:
        if not isinstance(base_ds_config, dict):
            raise TypeError("Expected string or dict for base_ds_config argument.")
    return base_ds_config


def get_ds_config_from_hparams(
    hparams: Dict[str, Any],
    model_dir: Union[pathlib.Path, str] = CURR_DIR,
) -> Dict[str, Any]:
    """Fetch and recursively merge the deepspeed config from the experiment config

    Follows the rules as described here:
    https://docs.determined.ai/latest/training/apis-howto/deepspeed/deepspeed.html#configuration
    Args:
        hparams (Dict):
            Hyperparameters dictionary
        model_dir (pathlib.Path):
            Base path for the Experiment Model
    Returns:
        The Deepspeed Configuration for this experiment following the overwriting rules
    """
    model_dir = pathlib.Path(model_dir)
    assert _defaults.CONFIG_KEY in hparams, (
        f"Expected to find {_defaults.CONFIG_KEY} in the Hyperparameters section. "
        f"Instead found {hparams}"
    )
    base_config_file_name = hparams[_defaults.CONFIG_KEY]
    base_ds_config = normalize_base_ds_config(base_config_file_name, model_dir=model_dir)
    overwrite_ds_config = hparams.get(_defaults.OVERWRITE_KEY, {})
    ds_config = merge_dicts(base_ds_config, overwrite_ds_config)
    return ds_config


def overwrite_deepspeed_config(
    base_ds_config: Union[str, Dict[str, Any]],
    source_ds_dict: Dict[str, Any],
    model_dir: pathlib.Path = CURR_DIR,
) -> Dict[str, Any]:
    """Overwrite a base_ds_config with values from a source_ds_dict.

    You can use source_ds_dict to overwrite leaf nodes of the base_ds_config.
    More precisely, we will iterate depth first into source_ds_dict and if a node corresponds to
    a leaf node of base_ds_config, we copy the node value over to base_ds_config.
    Arguments:
        base_ds_config (str or Dict): either a path to a DeepSpeed config file or a dictionary.
        source_ds_dict (Dict): dictionary with fields that we want to copy to base_ds_config
        model_dir (pathlib.Path): Base path for the Experiment Model
    Returns:
        The resulting dictionary when base_ds_config is overwritten with source_ds_dict.
    """
    normalized_base_ds_config = normalize_base_ds_config(base_ds_config, model_dir=model_dir)
    return merge_dicts(normalized_base_ds_config, source_ds_dict)


def get_hf_ds_config_path_from_args(args: List[str]) -> Optional[str]:
    for idx in range(len(args)):
        if args[idx] == "--deepspeed":
            ds_config_idx = idx + 1
            ds_config_path = args[ds_config_idx]
            return ds_config_path
    return None


def update_hf_args(args: List[str], ds_config_dict: Dict[str, Any]) -> List[str]:
    """
    Updates batch-size-related HF CLI args to be consistent with the values specified in the
    provided DeepSpeed config dictionary.

    Args:
        args: list of CLI arguments passed to the HF entrypoint
        ds_config_dict: the DeepSpeed configuration as a dictionary
    """
    hf_flag_to_ds_key = {
        "--per_device_train_batch_size": "train_micro_batch_size_per_gpu",
        "--gradient_accumulation_steps": "gradient_accumulation_steps",
    }
    # Overwrite CLI args
    args = copy.deepcopy(args)
    for idx in range(len(args)):
        if args[idx] in hf_flag_to_ds_key:
            ds_key = hf_flag_to_ds_key[args[idx]]
            overwrite_value = str(ds_config_dict[ds_key])
            if args[idx + 1] != overwrite_value:
                logging.warning(
                    f"Changing {args[idx]} from {args[idx +1]} to {overwrite_value} to match "
                    " the deespspeed config values."
                )
                args[idx + 1] = overwrite_value
            del hf_flag_to_ds_key[args[idx]]

    # Any remaining keys in hf_flag_to_ds_key were not provided as args to the HF CLI entrypoint,
    # but they must be added in explicitly, to avoid falling back to HF defaults.
    for hf_flag, ds_key in hf_flag_to_ds_key.items():
        hf_flag_value = str(ds_config_dict[ds_key])
        args.extend([hf_flag, hf_flag_value])
        logging.warning(
            f"Adding {hf_flag} {hf_flag_value} to HF CLI args to reflect overwrite values."
        )
    return args


def get_hf_args_with_overwrites(args: List[str], hparams: Dict[str, Any]) -> List[str]:
    """Updates the submitted HF CLI Args to account for overwrite values.

    Primarily intended as a helper function for Determined AI DeepSpeed (DS) Autotune which provides
    overwrite values through the `hparams["overwrite_deepspeed_args"]` which possibly include DS
    batch-size related arguments (`train_batch_size`, `train_micro_batch_size_per_gpu`, and
    `gradient_accumulation_steps`) which are in conflict with the corresponding HF CLI batch-size
    related arguments(`--per_device_train_batch_size` and `--gradient_accumulation_steps`). This
    function updates the HF CLI args to relect any such overwrite values. This process also requires
    overwriting the corresponding DS json file on-cluster.

    Args:
        args: the original HF CLI arguments
        hparams: hyperparameter dictionary generated through Determined AI

    Returns:
        args: updated HF CLI arguments
    """
    if _defaults.OVERWRITE_KEY not in hparams:
        logging.info(
            f"{_defaults.OVERWRITE_KEY} key not found in hparams, `get_hf_args_with_overwrites` "
            "is a no-op"
        )
        return args

    ds_config_path = get_hf_ds_config_path_from_args(args)
    assert ds_config_path is not None, "--deepspeed flag not found in HuggingFace args!"

    with open(ds_config_path, "r") as f:
        ds_config_dict = json.load(f)

    # Then merge all overwrites into the ds_config
    overwritten_ds_config_dict = merge_dicts(ds_config_dict, hparams[_defaults.OVERWRITE_KEY])

    # We need to actually overwrite the ds json config file, due to how HF processes args.
    # A file lock is required during both the writing and reading.
    with filelock.FileLock(ds_config_path + ".lock"):
        with open(ds_config_path, "w") as f:
            json.dump(overwritten_ds_config_dict, f)
        # Finally overwrite the CLI args
        args = update_hf_args(args, overwritten_ds_config_dict)

    return args