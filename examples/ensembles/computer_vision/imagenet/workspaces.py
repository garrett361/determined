import asyncio
import json
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Union, Sequence, Set, Optional

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

        self._idxs_to_delete_set = None
        self._project_trials_dict = defaultdict(list)

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
        """Creates a new Workspace, if it doesn't already exist."""
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
            raise ValueError(f"{self.workspace_name} Workspace not found!")
        return workspace_idx

    def _get_project_idxs(
        self, project_names: Optional[Union[Sequence[str], Set[str], str]] = None
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
            raise KeyError(f"Projects {project_names} not found in Workspace.")
        return project_idxs

    def _get_experiment_idxs(
        self, project_names: Optional[Union[Sequence[str], Set[str], str]] = None
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
            print(f"Project {project_name} already exists in the {self.workspace_name} Workspace.")
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
        self, project_names: Optional[Union[Sequence[str], Set[str], str]] = None
    ) -> List[Dict[str, Any]]:
        """Returns a list detailing all Experiments in the Workspace. If
        project_names is specified, only Experiments in the specified Projects are returned.
        """
        project_idxs = self._get_project_idxs(project_names)
        gather_fn_kwargs = (
            {"url": f"{self.master_url}/api/v1/experiments", "params": {"projectId": idx}}
            for idx in project_idxs
        )
        project_experiments = asyncio.run(
            self._gather(
                gather_fn=self._get_json_async,
                gather_fn_kwargs=gather_fn_kwargs,
                desc=f"Getting Experiments {'from' if project_names else ''} {project_names}",
            )
        )
        experiments = []
        for p in project_experiments:
            experiments += p["experiments"]
        return experiments

    def get_all_trials(
        self,
        project_names: Optional[Union[Sequence[str], Set[str], str]] = None,
        refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Returns a list detailing all Trials in the Workspace. If
        project_names is specified, only Trials in the specified Projects are returned. First call
        will download all relevant trial data and cache it. Subsequent calls will use the cached
        results, unless refresh is True.
        """
        if project_names is None:
            project_names = {wp["name"] for wp in self.get_all_projects()}
        if isinstance(project_names, str):
            project_names = {project_names}
        elif isinstance(project_names, Sequence):
            project_names = set(project_names)

        if refresh:
            self._project_trials_dict = defaultdict(list)

        required_project_names = project_names - set(self._project_trials_dict.keys())
        for name in required_project_names:
            experiments = self.get_all_experiments(name)
            experiment_idxs = [exp["id"] for exp in experiments]
            gather_fn_kwargs = [
                {"url": f"{self.master_url}/api/v1/experiments/{idx}/trials"}
                for idx in experiment_idxs
            ]
            exp_trials = asyncio.run(
                self._gather(
                    gather_fn=self._get_json_async,
                    gather_fn_kwargs=gather_fn_kwargs,
                    desc=f"Getting Trials from {name}",
                )
            )
            if not exp_trials:
                self._project_trials_dict[name] = []
            else:
                for e in exp_trials:
                    self._project_trials_dict[name] += e["trials"]
        trials = []
        for name in project_names:
            trials += self._project_trials_dict[name]
        return trials

    def get_trial_latest_val_results_dict(
        self,
        project_names: Optional[Union[Sequence[str], Set[str], str]] = None,
        refresh: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        """Returns a dict summarizing the latest validation for trial in the Workspace, indexed by
        trial ID.  If project_names is provided, only trials from those Projects will be returned,
        otherwise, all trials in the Workspace will be returned.
        """
        trial_results_dict = {}
        trials = self.get_all_trials(project_names, refresh)

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
        self,
        project_names: Optional[Union[Sequence[str], Set[str], str]] = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Returns a DataFrame summarizing the latest validation for trials in the Workspace,
        indexed by trial ID.  If project_names is provided, only trials from those Projects will be
        returned, otherwise, all trials in the Workspace will be returned.
        """
        trial_results_dict = self.get_trial_latest_val_results_dict(project_names, refresh)
        trial_results_df = pd.DataFrame.from_dict(trial_results_dict, orient="index")
        trial_results_df = trial_results_df[sorted(trial_results_df.columns)]
        return trial_results_df

    def _delete_experiment_idxs(self, experiment_idxs: Set[int], desc: str = "") -> None:
        gather_fn_kwargs = (
            {"url": f"{self.master_url}/api/v1/experiments/{idx}"} for idx in experiment_idxs
        )
        asyncio.run(
            self._gather(gather_fn=self._delete_async, gather_fn_kwargs=gather_fn_kwargs, desc=desc)
        )

    def delete_all_experiments(
        self, *, projects_to_delete_from: Union[Sequence[str], str], safe_mode: bool = True
    ) -> None:
        """Deletes all Experiments from the specified Projects in the Workspace.  Must be called
        twice to perform the deletion when safe_mode == True."""
        idxs_to_delete_set = set(self._get_experiment_idxs(projects_to_delete_from))
        if safe_mode and idxs_to_delete_set != self._idxs_to_delete_set:
            warnings.warn(
                f"This will delete {len(idxs_to_delete_set)} experiments."
                " Please run a second time to confirm deletion."
            )
            self._idxs_to_delete_set = idxs_to_delete_set
        else:
            print(f"Deleting {len(idxs_to_delete_set)} experiments.")
            self._delete_experiment_idxs(idxs_to_delete_set, desc="Deleting all Experiments")
            self._idxs_to_delete_set = None

    def delete_experiments_with_unvalidated_trials(
        self,
        *,
        projects_to_delete_from: Union[Sequence[str], str],
        safe_mode: bool = True,
        refresh: bool = False,
    ) -> None:
        """Deletes all finished Experiments which contain unvalidated Trials from the specified
        Projects in the Workspace.  Must be called twice to perform the deletion when
        safe_mode == True.
        """
        idxs_to_delete_set = set()
        trials = self.get_all_trials(projects_to_delete_from, refresh)
        for trial in trials:
            if (
                trial["state"] not in {"STATE_ACTIVE", "STATE_UNSPECIFIED"}
                and trial["latestValidation"] is None
            ):
                idxs_to_delete_set.add(trial["experimentId"])
        if safe_mode and self._idxs_to_delete_set != idxs_to_delete_set:
            warnings.warn(
                f"This will delete {len(idxs_to_delete_set)} experiments."
                " Please run a second time to confirm deletion."
            )
            self._idxs_to_delete_set = idxs_to_delete_set
        else:
            print(f"Deleting {len(idxs_to_delete_set)} experiments.")
            self._delete_experiment_idxs(
                idxs_to_delete_set, desc="Deleting unvalidated Experiments"
            )
            self._idxs_to_delete_set = None

    async def _gather(
        self, gather_fn: Callable, gather_fn_kwargs: List[Dict[str, Any]], desc: str = ""
    ) -> List[Dict[str, Any]]:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            output = await tqdm_asyncio.gather(
                *(gather_fn(session, **kwarg) for kwarg in gather_fn_kwargs),
                desc=desc,
            )
            return output

    async def _get_json_async(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        default_params = {"limit": REQ_LIMIT}
        if params is None:
            params = default_params
        else:
            params = {**default_params, **params}
        async with session.get(url, ssl=False, params=params) as response:
            return await response.json()

    async def _delete_async(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> None:
        await session.delete(url, ssl=False)
