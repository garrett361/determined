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
    "zero_optimization": {"stage": 3},
    "flops_profiler": FLOPS_PROFILER_CONFIG,
}

SINGLE_SEARCHER_CONFIG = {
    "name": "single",
    "max_length": MODEL_INFO_MAX_LENGTH,
    "metric": "placeholder",
}
