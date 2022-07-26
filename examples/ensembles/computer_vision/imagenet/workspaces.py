import asyncio
import json
from typing import Any, Callable, Dict, List, Union, Sequence, Set, Optional
import warnings

import aiohttp
import pandas as pd
import requests
from tqdm.asyncio import tqdm_asyncio

REQ_LIMIT = 2 ** 30


class Workspace:
    """
    A simple class for interacting with a Determined Workspace. Import ``nest_asyncio`` and run
    nest_asyncio.apply() before using in a Jupyter notebook.
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
        self._headers = self._get_headers()
        if create_workspace:
            self._create_workspace(workspace_name)
        self.workspace_idx = self._get_workspace_idx()

        self._last_delete_called = ""
        self._idxs_to_delete_set = set()

    def _session(self) -> requests.Session:
        s = requests.Session()
        s.headers = self._headers
        return s

    def _get_login_token(self) -> str:
        auth = json.dumps({"username": self.username, "password": self.password})
        login_url = f"{self.master_url}/login"
        try:
            response = requests.post(login_url, data=auth)
            token = response.json()["token"]
            return token
        except KeyError:
            print(f"Failed to log in to {self.master_url}. Is master running?")

    def _get_headers(self) -> Dict[str, str]:
        return dict(Cookie=f"auth={self.token}")

    def _create_workspace(self, workspace_name: str) -> None:
        """Creates a new workspace, if it doesn't already exist."""
        workspace_idx = None
        try:
            workspace_idx = self._get_workspace_idx()
        except ValueError:
            url = f"{self.master_url}/api/v1/workspaces"
            workspace_dict = {
                "name": workspace_name,
            }
            with self._session() as s:
                s.post(url, json=workspace_dict)
            print(f"Created workspace {workspace_name}.")
        if workspace_idx is not None:
            print(f"Workspace {workspace_name} already exists.")

    def _get_workspace_idx(self) -> int:
        url = f"{self.master_url}/api/v1/workspaces"
        with self._session() as s:
            response = s.get(url)
        data = json.loads(response.content)
        workspace_idx = None
        for workspace in data["workspaces"]:
            if workspace["name"] == self.workspace_name:
                workspace_idx = workspace["id"]
                break
        if workspace_idx is None:
            raise ValueError(f"{self.workspace_name} workspace not found!")
        return workspace_idx

    def _get_project_idxs(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> Set[int]:
        workspace_projects = self.get_all_projects()
        if project_names is None:
            project_names = {wp["name"] for wp in workspace_projects}
        elif isinstance(project_names, str):
            project_names = {project_names}
        else:
            project_names = set(project_names)
        project_idxs = set()
        for wp in workspace_projects:
            wp_name = wp["name"]
            if wp_name in project_names:
                project_idxs.add(wp["id"])
                project_names.remove(wp_name)
        if project_names:
            raise KeyError(f"Projects {project_names} not found in workspace.")
        return project_idxs

    def _get_experiment_idxs(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> List[int]:
        experiments = self.get_all_experiments(project_names)
        experiment_idxs = [e["id"] for e in experiments]
        return experiment_idxs

    def get_all_projects(self) -> List[Dict[str, Any]]:
        "Returns a list detailing all Projects in the Workspace."
        url = f"{self.master_url}/api/v1/workspaces/{self.workspace_idx}/projects"
        with self._session() as s:
            response = s.get(url, params={"limit": REQ_LIMIT})
        projects = json.loads(response.content)["projects"]
        return projects

    def get_all_project_names(self) -> List[str]:
        "Returns a list of all Project names in the Workspace."
        projects = self.get_all_projects()
        names = [p["name"] for p in projects]
        return names

    def create_project(self, project_name: str, description: str = "") -> None:
        """Creates a new project in the Workspace, if it doesn't already exist."""
        try:
            self._get_project_idxs(project_name)
            print(f"Project {project_name} already exists in the {self.workspace_name} workspace.")
            return
        except KeyError:
            url = f"{self.master_url}/api/v1/workspaces/{self.workspace_idx}/projects"
            project_dict = {
                "name": project_name,
                "description": description,
                "workspaceId": self.workspace_idx,
            }
            with self._session() as s:
                s.post(url, json=project_dict)

    def get_all_experiments(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> List[Dict[str, Any]]:
        """Returns a list detailing all Experiments in the Workspace. If
        project_names is specified, only Experiments in the specified Projects are returned.
        """
        project_idxs = self._get_project_idxs(project_names)
        urls = [f"{self.master_url}/api/v1/projects/{idx}/experiments" for idx in project_idxs]
        project_experiments = asyncio.run(
            self._gather_from_urls_async(
                urls, gather_fn=self._get_json_async, desc="Getting Experiments"
            )
        )
        experiments = []
        for p in project_experiments:
            experiments += p["experiments"]
        return experiments

    def get_all_trial_ids(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> List[int]:
        """Returns a list of all IDs for Trials in the Workspace. If project_names is specified,
        only Trials in the specified Projects are returned.
        """
        experiments = self.get_all_experiments(project_names=project_names)
        trial_ids = []
        for e in experiments:
            trial_ids += e["trialIds"]
        return trial_ids

    def get_all_trials(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> List[Dict[str, Any]]:
        """Returns a list detailing all Trials in the Workspace. If
        project_names is specified, only Trials in the specified Projects are returned.
        """
        experiments = self.get_all_experiments(project_names)
        experiment_idxs = [exp["id"] for exp in experiments]
        urls = [f"{self.master_url}/api/v1/experiments/{idx}/trials" for idx in experiment_idxs]
        exp_trials = asyncio.run(
            self._gather_from_urls_async(
                urls, gather_fn=self._get_json_async, desc="Getting Trials"
            )
        )
        trials = []
        for e in exp_trials:
            trials += e["trials"]
        return trials

    def get_trial_latest_val_results_dict(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Returns a dict summarizing the latest validation for trial in the Workspace, indexed by
        trial ID.  If project_names is provided, only trials from those Projects will be returned,
        otherwise, all trials in the Workspace will be returned.
        """
        trial_results_dict = {}
        trials = self.get_all_trials(project_names)

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
        """Returns a DataFrame summarizing the latest validation for trials in the Workspace,
        indexed by trial ID.  If project_names is provided, only trials from those Projects will be
        returned, otherwise, all trials in the Workspace will be returned.
        """
        trial_results_dict = self.get_trial_latest_val_results_dict(project_names)
        trial_results_df = pd.DataFrame.from_dict(trial_results_dict, orient="index")
        trial_results_df = trial_results_df[sorted(trial_results_df.columns)]
        return trial_results_df

    def delete_experiment_idxs(self, experiment_idxs: Set[int], desc: str = "") -> None:
        urls = [f"{self.master_url}/api/v1/experiments/{idx}" for idx in experiment_idxs]
        asyncio.run(self._gather_from_urls_async(urls, gather_fn=self._delete_async, desc=desc))

    def delete_all_experiments(self, projects_to_delete_from: Union[Sequence[str], str]) -> None:
        """Deletes all Experiments from the specified Projects in the Workspace.  Must be called
        twice to perform the deletion, as a safety measure."""
        self._idxs_to_delete_set = set(self._get_experiment_idxs(projects_to_delete_from))
        if self._last_delete_called != "delete_all_experiments":
            warnings.warn(
                f"This will delete {len(self._idxs_to_delete_set)} experiments."
                " Please run a second time to confirm deletion."
            )
            self._last_delete_called = "delete_all_experiments"
        else:
            self._last_delete_called = ""
            self.delete_experiment_idxs(self._idxs_to_delete_set, desc="Deleting all Experiments")
            self._idxs_to_delete_set = set()

    def delete_experiments_with_unvalidated_trials(
        self, projects_to_delete_from: Union[Sequence[str], str]
    ) -> None:
        """Deletes all Experiments which contain unvalidated Trials from the specified Projects in
        the Workspace.  Must be called twice to perform the deletion, as a safety measure.
        """
        if self._last_delete_called != "delete_experiments_with_unvalidated_trials":
            trials = self.get_all_trials(projects_to_delete_from)
            for trial in trials:
                if trial["latestValidation"] is None:
                    self._idxs_to_delete_set.add(trial["experimentId"])
            warnings.warn(
                f"This will delete {len(self._idxs_to_delete_set)} experiments."
                " Please run a second time to confirm deletion."
            )
            self._last_delete_called = "delete_experiments_with_unvalidated_trials"
        else:
            self._last_delete_called = ""
            self.delete_experiment_idxs(
                self._idxs_to_delete_set, desc="Deleting unvalidated Experiments"
            )
            self._idxs_to_delete_set = set()

    async def _gather_from_urls_async(
        self, urls: List[str], gather_fn: Callable, desc: str = ""
    ) -> List[Dict[str, Any]]:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            output = await tqdm_asyncio.gather(
                *(gather_fn(url, session) for url in urls), desc=desc
            )
            return output

    async def _get_json_async(self, url: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        async with session.get(url, ssl=False, params={"limit": REQ_LIMIT}) as response:
            return await response.json()

    async def _delete_async(self, url: str, session: aiohttp.ClientSession) -> None:
        await session.delete(url, ssl=False)
