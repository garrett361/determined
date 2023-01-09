MODEL_INFO_MAX_LENGTH = 5
WORKDIR_PATH = "/run/determined/workdir/"
OUTPUT_FILE_PATH = WORKDIR_PATH + "flops_profiler_output.txt"

FLOPS_PROFILER_CONFIG = {
    "enabled": True,
    "profile_step": MODEL_INFO_MAX_LENGTH - 1,
    "module_depth": -1,
    "top_modules": 10,
    "detailed": True,
    "output_file": OUTPUT_FILE_PATH,
}

MODEL_INFO_PROFILING_DS_CONFIG = {
    "train_micro_batch_size_per_gpu": 1,
    "zero_optimization": {
        "stage": 0
    },  # DS set the stage to 3; not sure why? See DEFAULT_MIN_MEM_CONFIG
    "flops_profiler": FLOPS_PROFILER_CONFIG,
}

# TODO: Should remove all references to SINGLE_SEARCHER_CONFIG and make max_length
# available for user to define.
SINGLE_SEARCHER_CONFIG = {
    "name": "single",
    "max_length": MODEL_INFO_MAX_LENGTH,
    "metric": "placeholder",
}
