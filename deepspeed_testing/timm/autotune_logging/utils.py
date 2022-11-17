import json
import os
import pathlib
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


class DSAutotuningResults:
    """Class for extracting results from DS autotuning dirs."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self.base_path = base_path

        self.results_base_path = self.base_path.joinpath("autotuning_results")

        self.model_info = self._get_model_info()
        self.results_dirs = self._get_results_dirs()

    def _get_model_info(self) -> Dict[str, Any]:
        model_info_path = self.results_base_path.joinpath("profile_model_info/model_info.json")
        model_info = self._get_dict_from_json_path(model_info_path)
        return model_info

    def _get_results_dirs(self) -> List[pathlib.Path]:
        results_dirs = []
        for d in os.listdir(self.results_base_path):
            d = self.results_base_path.joinpath(d)
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

                ds_config = self._get_dict_from_json_path(path.joinpath("ds_config.json"))
                hp_config["ds_config"] = ds_config

                hp_config = upper_case_dict_key(hp_config, "type")
                hp_config_list.append(hp_config)
        return hp_config_list

    def get_grid_search_config(
        self,
        workspace_name: str,
        project_name: str,
        exp_name: str,
        entrypoint: str,
        append_to_name: str = ".results",
    ) -> Dict[str, Any]:
        grid_search_config = {
            "entrypoint": entrypoint,
            "name": exp_name + append_to_name,
            "workspace": workspace_name,
            "project": project_name,
            "max_restarts": 1,
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
        grid_search_config["searcher"]["metric"] = all_hp_dicts[0]["ds_config"]["autotuning"][
            "metric"
        ]
        grid_search_config["searcher"]["smaller_is_better"] = (
            grid_search_config["searcher"]["metric"] == "latency"
        )

        grid_search_config["hyperparameters"] = {
            "results": {"type": "categorical", "vals": all_hp_dicts}
        }
        return grid_search_config
