import json
from typing import Any, Dict, List, Union, Sequence, Set, Optional
import pandas as pd
import requests

GET_LIMIT = 2 ** 16


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
            self._create_workspace(workspace_name)
        self.workspace_id = self._get_workspace_id()

    def get_all_projects(self) -> List[Dict[str, Any]]:
        url = f"{self.master_url}/api/v1/workspaces/{self.workspace_id}/projects"
        response = requests.get(url, params={"limit": GET_LIMIT}, headers=self.py_request_headers)
        projects = json.loads(response.content)["projects"]
        return projects

    def get_all_project_names(self) -> List[str]:
        projects = self.get_all_projects()
        names = [p["name"] for p in projects]
        return names

    def create_project(self, project_name: str, description: str = "") -> None:
        """Creates a new project in the workspace, if it doesn't already exist."""
        if self._get_project_ids(project_name):
            print(f"Project {project_name} already exists in the {self.workspace_name} workspace.")
            return
        url = f"{self.master_url}/api/v1/workspaces/{self.workspace_id}/projects"
        project_dict = {
            "name": project_name,
            "description": description,
            "workspaceId": self.workspace_id,
        }
        requests.post(url, headers=self.py_request_headers, json=project_dict)

    def get_experiments(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> List[Dict[str, Any]]:
        project_ids = self._get_project_ids(project_names)
        exps = []
        for pid in project_ids:
            url = f"{self.master_url}/api/v1/projects/{pid}/experiments"
            response = requests.get(
                url, params={"limit": GET_LIMIT}, headers=self.py_request_headers
            )
            pid_exp = json.loads(response.content)["experiments"]
            exps += pid_exp
        return exps

    def get_trials(
        self,
        project_names: Optional[Union[Sequence[str], str]] = None,
    ) -> List[Dict[str, Any]]:
        experiments = self.get_experiments(project_names)
        experiment_ids = [exp["id"] for exp in experiments]

        trials = []
        for idx in experiment_ids:
            url = f"{self.master_url}/api/v1/experiments/{idx}/trials"
            response = requests.get(url, headers=self.py_request_headers)
            data = json.loads(response.content)
            for trial in data["trials"]:
                trials.append(trial)
        return trials

    def get_trial_latest_val_results_dict(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Returns a dict summarizing the latest validation for trial in the workspace, indexed by
        trial ID.  If project_names is provided, only trials from those projects will be returned,
        otherwise, all trials in the workspace will be returned.
        """
        trial_results_dict = {}
        trials = self.get_trials(project_names)
        for trial in trials:
            if trial["latestValidation"] is None:
                continue
            trial_results = trial["latestValidation"]["metrics"]
            trial_results["wall_clock_time"] = trial["wallClockTime"]
            trial_results["experiment_id"] = trial["experimentId"]
            idx = trial["id"]
            trial_results_dict[idx] = trial_results
        return trial_results_dict

    def get_trial_latest_val_results_df(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> pd.DataFrame:
        """Returns a DataFrame summarizing the latest validation for trials in the workspace,
        indexed by trial ID.  If project_names is provided, only trials from those projects will be
        returned, otherwise, all trials in the workspace will be returned.
        """
        trial_results_dict = self.get_trial_latest_val_results_dict(project_names)
        trial_results_df = pd.DataFrame.from_dict(trial_results_dict, orient="index")
        trial_results_df = trial_results_df[sorted(trial_results_df.columns)]
        return trial_results_df

    def delete_all_experiments(
        self, project_names: Union[Sequence[str], str], true_to_confirm_delete: bool = False
    ) -> None:
        if not true_to_confirm_delete:
            raise ValueError(
                f"Please confirm deletion of all experiments in {project_names} via the"
                f"true_to_confirm_delete flag."
            )
        experiment_ids = self._get_experiment_ids(project_names)
        for idx in experiment_ids:
            url = f"{self.master_url}/api/v1/experiments/{idx}"
            requests.delete(url, headers=self.py_request_headers)

    def _get_login_token(self) -> str:
        auth = json.dumps({"username": self.username, "password": self.password})
        login_url = f"{self.master_url}/login"
        try:
            response = requests.post(login_url, data=auth)
        except KeyError:
            print(f"Failed to log in to {self.master_url}. Is master running?")
            return
        token = response.json()["token"]
        return token

    def _get_py_request_headers(self) -> Dict[str, str]:
        return dict(Cookie=f"auth={self.token}")

    def _create_workspace(self, workspace_name: str) -> None:
        """Creates a new workspace, if it doesn't already exist."""
        workspace_id = None
        try:
            workspace_id = self._get_workspace_id()
        except ValueError:
            url = f"{self.master_url}/api/v1/workspaces"
            workspace_dict = {
                "name": workspace_name,
            }
            requests.post(url, headers=self.py_request_headers, json=workspace_dict)
            print(f"Created workspace {workspace_name}.")
        if workspace_id is not None:
            print(f"Workspace {workspace_name} already exists.")

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

    def _get_project_ids(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> Set[int]:
        workspace_projects = self.get_all_projects()
        if project_names is None:
            project_names = {wp["name"] for wp in workspace_projects}
        elif isinstance(project_names, str):
            project_names = {project_names}
        else:
            project_names = set(project_names)
        project_ids = set()
        for wp in workspace_projects:
            wp_name = wp["name"]
            if wp_name in project_names:
                project_ids.add(wp["id"])
                project_names.remove(wp_name)
        if project_names:
            raise KeyError(f"Projects {project_names} not found in workspace.")
        return project_ids

    def _get_experiment_ids(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> List[Dict[str, Any]]:
        experiments = self.get_experiments(project_names)
        experiment_ids = [e["id"] for e in experiments]
        return experiment_ids
