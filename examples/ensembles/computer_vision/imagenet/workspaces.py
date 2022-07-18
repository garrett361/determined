import asyncio

import aiohttp
import json
from typing import Any, Dict, List, Union, Sequence, Set, Optional
import pandas as pd
import requests

REQ_LIMIT = 2 ** 30


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
        self._headers = self._get_headers()
        if create_workspace:
            self._create_workspace(workspace_name)
        self.workspace_id = self._get_workspace_id()

    def _session(self) -> requests.Session:
        s = requests.Session()
        s.headers = self._headers
        return s

    def get_all_projects(self) -> List[Dict[str, Any]]:
        url = f"{self.master_url}/api/v1/workspaces/{self.workspace_id}/projects"
        with self._session() as s:
            response = s.get(url, params={"limit": REQ_LIMIT})
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
        with self._session() as s:
            s.post(url, json=project_dict)

    def get_experiments(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> List[Dict[str, Any]]:
        project_ids = self._get_project_ids(project_names)
        exps = []
        with self._session() as s:
            for pid in project_ids:
                url = f"{self.master_url}/api/v1/projects/{pid}/experiments"
                response = s.get(url, params={"limit": REQ_LIMIT})
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
        with self._session() as s:
            for idx in experiment_ids:
                url = f"{self.master_url}/api/v1/experiments/{idx}/trials"
                response = s.get(url)
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
        with self._session() as s:
            for idx in experiment_ids:
                url = f"{self.master_url}/api/v1/experiments/{idx}"
                s.delete(url)

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

    def _get_headers(self) -> Dict[str, str]:
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
            with self._session() as s:
                s.post(url, json=workspace_dict)
            print(f"Created workspace {workspace_name}.")
        if workspace_id is not None:
            print(f"Workspace {workspace_name} already exists.")

    def _get_workspace_id(self) -> int:
        url = f"{self.master_url}/api/v1/workspaces"
        with self._session() as s:
            response = s.get(url)
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


class WorkspaceAIO:
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
        self._headers = self._get_headers()
        if create_workspace:
            self._create_workspace(workspace_name)
        self.workspace_id = self._get_workspace_id()
        self.delete_confirmed = False

        self.PARALLEL_REQUESTS = 100
        self.LIMIT_PER_HOST = 100
        self.LIMIT = 0
        self.TTL_DNS_CACHE = 300

    def _async_session(self) -> aiohttp.ClientSession:
        s = aiohttp.ClientSession(headers=self._headers)
        return s

    def _session(self) -> requests.Session:
        s = requests.Session()
        s.headers = self._headers
        return s

    def get_all_projects(self) -> List[Dict[str, Any]]:
        url = f"{self.master_url}/api/v1/workspaces/{self.workspace_id}/projects"
        with self._session() as s:
            response = s.get(url, params={"limit": REQ_LIMIT})
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
        with self._session() as s:
            s.post(url, json=project_dict)

    def get_experiments(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> List[Dict[str, Any]]:
        project_ids = self._get_project_ids(project_names)
        exps = []
        with self._session() as s:
            for pid in project_ids:
                url = f"{self.master_url}/api/v1/projects/{pid}/experiments"
                response = s.get(url, params={"limit": REQ_LIMIT})
                pid_exp = json.loads(response.content)["experiments"]
                exps += pid_exp
        return exps

    async def _get_all_experiments_async(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> List[Dict[str, Any]]:
        project_idxs = self._get_project_ids(project_names)
        urls = [f"{self.master_url}/api/v1/projects/{idx}/experiments" for idx in project_idxs]
        exps = []
        await self._async_get_json(urls, exps)
        return exps

    async def _get_all_experiment_trials_async(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> List[Dict[str, Any]]:
        """Returns a list of all trials associated with each experiment in workspace. If
        project_names is not None, only experiments in the specified projects are returned.
        """
        experiments = self.get_experiments(project_names)
        experiment_idxs = [exp["id"] for exp in experiments]
        urls = [f"{self.master_url}/api/v1/experiments/{idx}/trials" for idx in experiment_idxs]
        exp_trials = []
        await self._async_get_json(urls, exp_trials)
        return exp_trials

    def get_trial_latest_val_results_dict(
        self, project_names: Optional[Union[Sequence[str], str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Returns a dict summarizing the latest validation for trial in the workspace, indexed by
        trial ID.  If project_names is provided, only trials from those projects will be returned,
        otherwise, all trials in the workspace will be returned.
        """
        trial_results_dict = {}
        exp_trials = asyncio.run(self._get_all_experiment_trials_async(project_names))

        for exp in exp_trials:
            for trial in exp["trials"]:
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

    def delete_all_experiments(self, projects_to_delete: Union[Sequence[str], str]) -> None:
        if not self.delete_confirmed:
            raise ValueError("Please run a second time to confirm deletion.")
            self.delete_confirmed = True
        self.delete_confirmed = Falses
        experiment_ids = self._get_experiment_ids(projects_to_delete)
        with self._session() as s:
            for idx in experiment_ids:
                url = f"{self.master_url}/api/v1/experiments/{idx}"
                s.delete(url)

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

    def _get_headers(self) -> Dict[str, str]:
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
            with self._session() as s:
                s.post(url, json=workspace_dict)
            print(f"Created workspace {workspace_name}.")
        if workspace_id is not None:
            print(f"Workspace {workspace_name} already exists.")

    def _get_workspace_id(self) -> int:
        url = f"{self.master_url}/api/v1/workspaces"
        with self._session() as s:
            response = s.get(url)
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

    async def _async_get_json(self, urls, target_list):
        """Based on https://blog.jonlu.ca/posts/async-python-http"""
        conn = aiohttp.TCPConnector(
            limit_per_host=self.LIMIT_PER_HOST, limit=self.LIMIT, ttl_dns_cache=self.TTL_DNS_CACHE
        )

        async def gather_with_concurrency():
            semaphore = asyncio.Semaphore(self.PARALLEL_REQUESTS)
            session = aiohttp.ClientSession(connector=conn, headers=self._headers)

            async def get(url):
                async with semaphore:
                    async with session.get(url, ssl=False) as response:
                        obj = json.loads(await response.read())
                        target_list.append(obj)

            await asyncio.gather(*(get(url) for url in urls))
            await session.close()

        asyncio.run(gather_with_concurrency())
        conn.close()
