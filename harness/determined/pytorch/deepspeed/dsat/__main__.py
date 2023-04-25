import argparse
import os
import pathlib
import pickle
import tempfile

from determined.experimental import client
from determined.pytorch.deepspeed.dsat import _defaults, _utils
from determined.util import merge_dicts


def parse_args() -> argparse.Namespace:
    # TODO: Allow for additional includes args to be specified, as in the CLI.
    parser = argparse.ArgumentParser(description="DS Autotuning")
    parser.add_argument("config_path")
    parser.add_argument("model_dir")
    parser.add_argument("-i", "--include", type=str, nargs="+")

    parser.add_argument("-s", "--search-runner-config", type=str)
    parser.add_argument("-t", "--tuner-type", type=str, default="random")
    parser.add_argument("-n", "--num-trials", type=int, default=50)

    # DS-specific args.
    parser.add_argument("-ss", "--start_profile-step", type=int, default=3)
    parser.add_argument("-es", "--end-profile-step", type=int, default=5)
    parser.add_argument("-ds", "--deepspeed-config", type=str, default="deepspeed_config")
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="throughput",
        choices=["throughput", "FLOPS_per_gpu", "forward", "backward", "latency"],
    )

    args = parser.parse_args()

    # Convert the paths to absolute paths
    args.config_path = os.path.abspath(args.config_path)
    args.model_dir = os.path.abspath(args.model_dir)
    args.include = [os.path.abspath(p) for p in args.include] if args.include is not None else []

    assert (
        args.tuner_type in _defaults.ALL_SEARCH_METHOD_CLASSES
    ), f"tuner-type must be one of {list(_defaults.ALL_SEARCH_METHOD_CLASSES)}, not {args.tuner_type}"

    return args

    parser.add_argument("config_path")
    parser.add_argument("model_dir")
    args = parser.parse_args()

    assert (
        args.tuner_type in _defaults.ALL_SEARCH_METHOD_CLASSES
    ), f"tuner-type must be one of {list(_defaults.ALL_SEARCH_METHOD_CLASSES)}, not {args.tuner_type}"

    return args


def run_autotuning(args: argparse.Namespace) -> None:
    # Build the default SearchRunner's config from the submitted config. The original config yaml file
    # is added as an include and is reimported by the SearchRunner later.

    config = _utils.get_search_runner_config_from_args(args)
    # TODO: early sanity check the submitted config.

    # Create empty tempdir as the model_dir and upload everything else as an includes in order to
    # preserve the top-level model_dir structure inside the SearchRunner's container.

    with tempfile.TemporaryDirectory() as temp_dir:
        # Upload the args, which will be used by the search runner on-cluster.
        args_path = pathlib.Path(temp_dir).joinpath("args.pkl")
        with args_path.open("wb") as f:
            pickle.dump(args, f)
        includes = [args.model_dir, args.config_path]
        client.create_experiment(config=config, model_dir=temp_dir, includes=includes)


if __name__ == "__main__":
    args = parse_args()
    run_autotuning(args)
