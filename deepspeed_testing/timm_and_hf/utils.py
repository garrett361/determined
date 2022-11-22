import json
import os
import pathlib
import re
from typing import Any, Dict, List


def upper_case_dict_key(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    upper_d = {}
    for k, v in d.items():
        new_k = k.upper() if key == k else k
        if isinstance(v, dict):
            upper_d[new_k] = upper_case_dict_key(v, key)
        else:
            upper_d[new_k] = v
    return upper_d


def get_decimal_numbers_in_line(line: str) -> float:
    num_str = re.search(r"\b\d*\.\d+\b", line).group()
    num = float(num_str)
    return num


class DSAutotuningResults:
    """Class for extracting results from DS autotuning dirs."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path

        self.results_path = self.path.joinpath("autotuning_results")

        self.model_info = self._get_model_info()
        self.results_dirs = self._get_results_dirs()

    def _get_model_info(self) -> Dict[str, Any]:
        model_info_path = self.results_path.joinpath("profile_model_info/model_info.json")
        model_info = self._get_dict_from_json_path(model_info_path)
        return model_info

    def _get_results_dirs(self) -> List[pathlib.Path]:
        results_dirs = []
        for d in os.listdir(self.results_path):
            d = self.results_path.joinpath(d)
            if d.is_dir() and d.stem[0] == "z":
                results_dirs.append(d)
        return results_dirs

    def _get_dict_from_json_path(self, path: pathlib.Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d

    def _get_hp_config_list_from_results_dir(self) -> List[Dict[str, Any]]:
        hp_config_list = []
        for path in self.results_dirs:
            if os.path.exists(path.joinpath("metrics.json")):
                hp_config = {}
                metrics_dict = self._get_dict_from_json_path(path.joinpath("metrics.json"))
                hp_config["metrics"] = metrics_dict

                exp_config = self._get_dict_from_json_path(path.joinpath("exp.json"))
                hp_config["exp_config"] = exp_config

                hp_config = upper_case_dict_key(hp_config, "type")
                hp_config_list.append(hp_config)
        return hp_config_list

    def get_grid_search_config(
        self,
        workspace_name: str,
        project_name: str,
        exp_name: str,
        model_name: str,
        entrypoint: str,
        append_to_name: str = ".results",
    ) -> Dict[str, Any]:
        grid_search_config = {
            "entrypoint": entrypoint,
            "name": exp_name + append_to_name,
            "workspace": workspace_name,
            "project": project_name,
            "max_restarts": 5,
            "resources": {"slots_per_trial": 0},
            "searcher": {
                "name": "grid",
                "max_length": 0,
                "metric": None,
                "smaller_is_better": None,
            },
            "hyperparameters": None,
        }

        all_hp_dicts = self._get_hp_config_list_from_results_dir()

        # Dynamically set some fields in the base config
        grid_search_config["searcher"]["metric"] = all_hp_dicts[0]["exp_config"]["ds_config"][
            "autotuning"
        ]["metric"]
        grid_search_config["searcher"]["smaller_is_better"] = (
            grid_search_config["searcher"]["metric"] == "latency"
        )

        # Add hp fields
        model_info_path = self.path.joinpath(
            "autotuning_results/profile_model_info/model_info.json"
        )
        model_info = self._get_dict_from_json_path(model_info_path)
        hp_dict = {
            "model_name": model_name,
            "model_info": model_info,
            "results": {"type": "categorical", "vals": all_hp_dicts},
        }
        grid_search_config["hyperparameters"] = hp_dict

        return grid_search_config


class DSProfilerResults:
    """Class for extracting results from DS profiler output."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path

    def _get_results_dict_from_path(self) -> Dict[str, float]:
        naming_map = {
            "iter latency": "latency_s",
            "FLOPS per GPU": "FLOPS_per_gpu_corrected",
            "samples/second": "throughput_s",
        }
        results_dict = {}
        with open(self.path, "r") as output:
            for line in output:
                line = line.strip()
                for name, metric in naming_map.items():
                    if line.startswith(name):
                        results_dict[metric] = get_decimal_numbers_in_line(line)
        return results_dict

    def get_config(
        self,
        workspace_name: str,
        project_name: str,
        exp_name: str,
        model_name: str,
        entrypoint: str,
        append_to_name: str = ".results",
    ) -> Dict[str, Any]:
        config = {
            "entrypoint": entrypoint,
            "name": exp_name + append_to_name,
            "workspace": workspace_name,
            "project": project_name,
            "max_restarts": 5,
            "resources": {"slots_per_trial": 0},
            "searcher": {
                "name": "single",
                "max_length": 0,
                "metric": "none",
            },
            "hyperparameters": None,
        }

        results_dict = self._get_results_dict_from_path()

        config["hyperparameters"] = {"results": results_dict, "profiled": True}

        return config