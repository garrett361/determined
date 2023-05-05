import argparse
import os
import pathlib
import pickle
import tempfile

from determined.experimental import client
from determined.pytorch.dsat import _defaults, _utils


def parse_args() -> argparse.Namespace:
    # TODO: Allow for additional includes args to be specified, as in the CLI.
    parser = _utils.get_parser()
    args = parser.parse_args()

    # Convert the paths to absolute paths
    args.config_path = os.path.abspath(args.config_path)
    args.model_dir = os.path.abspath(args.model_dir)
    args.include = [os.path.abspath(p) for p in args.include] if args.include is not None else []

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
        args_path = pathlib.Path(temp_dir).joinpath(_defaults.ARGS_PKL_PATH)
        with args_path.open("wb") as f:
            pickle.dump(args, f)
        includes = [args.model_dir, args.config_path] + args.include
        client.create_experiment(config=config, model_dir=temp_dir, includes=includes)


if __name__ == "__main__":
    args = parse_args()
    run_autotuning(args)
