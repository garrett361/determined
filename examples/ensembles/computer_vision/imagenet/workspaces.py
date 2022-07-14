import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Sequence, Set, Optional
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt


class Workspace:
    """
    A simple class for interacting with a Determined Workspace.
    """

    def __init__(
        self,
        workspace_name: str,
        master_url: str = "localhost:8080",
        username: str = "determined",
        password: str = "",
        create_workspace: bool = False,
    ) -> None:
        self.workspace_name = workspace_name
        self.master_url = master_url
        self.username = username
        self.password = password
        self.token = self._get_login_token()
        self.py_request_headers = self._get_py_request_headers()
        if create_workspace:
            pass

    def _get_login_token(self) -> str:
        auth = json.dumps({"username": self.username, "password": self.password})
        login_url = f"{self.master_url}/login"
        try:
            response = requests.post(login_url, data=auth)
        except KeyError:
            print(f"Failed to log in to {self.master_url}. Is Determined running?")
            return
        token = response.json()["token"]
        return token

    def _get_py_request_headers(self) -> Dict[str, str]:
        return dict(Cookie=f"auth={self.token}")

    def _get_workspace_id(self) -> int:
        url = f"{self.master_url}/api/v1/workspaces"
        response = requests.get(url, headers=self.py_request_headers)
        data = json.loads(response.content)
        workspace_id = None
        for workspace in data["workspaces"]:
            if workspace["name"] == self.workspace_name:
                workspace_id = workspace["id"]
                break
        if workspace_id is None:
            raise ValueError(f"{self.workspace_name} workspace not found!")
        return workspace_id

    def get_workspace_projects(self) -> List[Dict[str, Any]]:
        workspace_id = self._get_workspace_id()
        url = f"{self.master_url}/api/v1/workspaces/{workspace_id}/projects"
        response = requests.get(url, headers=self.py_request_headers)
        projects = json.loads(response.content)["projects"]
        return projects

    def _get_project_ids_from_workspace(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> Set[int]:
        workspace_projects = self.get_workspace_projects()
        if project_names is None:
            project_names = {wp["name"] for wp in workspace_projects}
        elif isinstance(project_names, str):
            project_names = {project_names}
        else:
            project_names = set(project_names)
        project_ids = set()
        for project in workspace_projects:
            if project["name"] in project_names:
                project_ids.add(project["id"])
        return project_ids

    def get_exps_from_workspace(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> List[Dict[str, Any]]:
        project_ids = self._get_project_ids_from_workspace(project_names)
        exps = []
        for pid in project_ids:
            url = f"{self.master_url}/api/v1/projects/{pid}/experiments"
            response = requests.get(url, headers=self.py_request_headers)
            pid_exp = json.loads(response.content)["experiments"]
            exps += pid_exp
        return exps

    def get_trials_from_workspace(
        self,
        project_names: Optional[Union[Sequence[str], str]] = None,
    ) -> List[Dict[str, Any]]:
        experiments = self.get_exps_from_workspace(project_names)
        experiment_ids = [exp["id"] for exp in experiments]

        trials = []
        for idx in experiment_ids:
            url = f"{self.master_url}/api/v1/experiments/{idx}/trials"
            response = requests.get(url, headers=self.py_request_headers)
            data = json.loads(response.content)
            for trial in data["trials"]:
                trials.append(trial)
        return trials

    def get_trial_results_dict_from_workspace(
        self,
        project_names: Optional[Union[Sequence[str], str]] = None,
        validated: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        """Returns a dict of all trial results, indexed by trial ID.  If project_names is provided,
        only trials from those projects will be returned, otherwise, all trials in the workspace
        will be returned. If validated is True, only validated trials will be returned.
        """
        trial_results_dict = {}
        trials = self.get_trials_from_workspace(project_names)
        for trial in trials:
            trial_results = {}
            if trial["latestValidation"] is not None:
                trial_results = trial["latestValidation"]["metrics"]
            elif validated:
                continue
            trial_results["wall_clock_time"] = trial["wallClockTime"]
            trial_results["experiment_id"] = trial["experimentId"]
            idx = trial["id"]
            trial_results_dict[idx] = trial_results
        return trial_results_dict

    def get_trial_results_df_from_workspace(
        self,
        project_names: Optional[Union[Sequence[str], str]] = None,
        validated: bool = False,
    ) -> pd.DataFrame:
        """Returns a DataFrame of all trial results, indexed by trial ID.  If project_names is
        provided, only trials from those projects will be returned, otherwise, all trials in the
        workspace will be returned. If validated is True, only validated trials will be returned.
        """
        trial_results_dict = self.get_trial_results_dict_from_workspace(project_names, validated)
        trial_results_df = pd.DataFrame.from_dict(trial_results_dict, orient="index")
        trial_results_df = trial_results_df[sorted(trial_results_df.columns)]
        return trial_results_df
