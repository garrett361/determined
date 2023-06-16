import asyncio
import json
import os
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Set, Union

import aiohttp
import pandas as pd
import requests
from tqdm.asyncio import tqdm_asyncio
from workspaces._utils import get_flattened_dict


# For use in a notebook https://stackoverflow.com/a/39662359
def is_notebook() -> bool:
    try:
        # get_ipython is defined when in notebook environments.
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False  # Probably standard Python interpreter


if is_notebook():
    import nest_asyncio

    nest_asyncio.apply()


REQ_LIMIT = 2**30


class Workspace:
    """
    A simple class for interacting with a Determined Workspace. Import ``nest_asyncio`` and run
    nest_asyncio.apply() before using in a Jupyter notebook.
    """

    def __init__(
        self,
        workspace_name: str,
        username: str = "determined",
        password: str = "",
        create_workspace: bool = False,
        master_url: Optional[str] = None,
    ) -> None:
        self.workspace_name = workspace_name
        self.username = username
        self.password = password
        if master_url is None:
            master_url = os.environ.get("DET_MASTER", "http://127.0.0.1:8080")
        self.master_url = master_url

        self.token = self._get_login_token()
        self._headers = self._get_headers()
        if create_workspace:
            self._create_workspace(workspace_name)
        self.workspace_idx = self._get_workspace_idx()

        self._exp_idxs_to_delete_set = None
        self._trial_idxs_to_kill_set = None
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
        """Returns a list detailing all Projects in the Workspace."""
        url = f"{self.master_url}/api/v1/workspaces/{self.workspace_idx}/projects"
        with self._session() as s:
            response = s.get(url, params={"limit": REQ_LIMIT})
        projects = json.loads(response.content)["projects"]
        return projects

    def get_all_project_names(self) -> List[str]:
        """Returns a list of all Project names in the Workspace."""
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
            project_names = set(self.get_all_project_names())
        elif isinstance(project_names, str):
            project_names = {project_names}
        elif isinstance(project_names, Sequence):
            project_names = set(project_names)

        if refresh:
            self._project_trials_dict = defaultdict(list)

        required_project_names = project_names - set(self._project_trials_dict)
        for name in required_project_names:
            experiments = self.get_all_experiments(name)
            experiment_idxs = [exp["id"] for exp in experiments]
            gather_fn_kwargs = (
                {"url": f"{self.master_url}/api/v1/experiments/{idx}/trials"}
                for idx in experiment_idxs
            )
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

    def get_trial_best_val_results_dict(
        self,
        project_names: Optional[Union[Sequence[str], Set[str], str]] = None,
        refresh: bool = False,
        only_get_completed: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """Returns a dict summarizing the best validation for trial in the Workspace, indexed by
        trial ID.  If project_names is provided, only trials from those Projects will be returned,
        otherwise, all trials in the Workspace will be returned.
        """
        trial_results_dict = {}
        trials = self.get_all_trials(project_names, refresh)

        for trial in trials:
            if trial["bestValidation"] is None:
                continue
            if only_get_completed and trial["state"] != "STATE_COMPLETED":
                continue
            best_val_metrics_dict = trial["bestValidation"]["metrics"]["avgMetrics"]
            flattened_hparam_dict = get_flattened_dict(trial["hparams"])
            trial_results = {
                **best_val_metrics_dict,
                **flattened_hparam_dict,
                "wall_clock_time": trial["wallClockTime"],
                "experiment_id": trial["experimentId"],
            }
            idx = trial["id"]
            trial_results_dict[idx] = trial_results
        return trial_results_dict

    def get_trial_best_val_results_df(
        self,
        project_names: Optional[Union[Sequence[str], Set[str], str]] = None,
        refresh: bool = False,
        only_get_completed: bool = True,
    ) -> pd.DataFrame:
        """Returns a DataFrame summarizing the best validation for trials in the Workspace,
        indexed by trial ID.  If project_names is provided, only trials from those Projects will be
        returned, otherwise, all trials in the Workspace will be returned.
        """
        trial_results_dict = self.get_trial_best_val_results_dict(
            project_names, refresh, only_get_completed
        )
        trial_results_df = pd.DataFrame.from_dict(trial_results_dict, orient="index")
        trial_results_df = trial_results_df[sorted(trial_results_df.columns)]
        return trial_results_df

    def get_all_trials_df(
        self,
        project_names: Optional[Union[Sequence[str], Set[str], str]] = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Returns a dict summarizing the best validation for trial in the Workspace, indexed by
        trial ID.  If project_names is provided, only trials from those Projects will be returned,
        otherwise, all trials in the Workspace will be returned.
        """
        trials = self.get_all_trials(project_names, refresh)
        trials_df = pd.DataFrame.from_dict(trials)

        return trials_df

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
        if safe_mode and idxs_to_delete_set != self._exp_idxs_to_delete_set:
            warnings.warn(
                f"This will delete {len(idxs_to_delete_set)} experiments."
                " Please run a second time to confirm deletion."
            )
            self._exp_idxs_to_delete_set = idxs_to_delete_set
        else:
            print(f"Deleting {len(idxs_to_delete_set)} experiments.")
            self._delete_experiment_idxs(idxs_to_delete_set, desc="Deleting all Experiments")
            self._exp_idxs_to_delete_set = None

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
                and trial["bestValidation"] is None
            ):
                idxs_to_delete_set.add(trial["experimentId"])
        if safe_mode and self._exp_idxs_to_delete_set != idxs_to_delete_set:
            warnings.warn(
                f"This will delete {len(idxs_to_delete_set)} experiments."
                " Please run a second time to confirm deletion."
            )
            self._exp_idxs_to_delete_set = idxs_to_delete_set
        else:
            print(f"Deleting {len(idxs_to_delete_set)} experiments.")
            self._delete_experiment_idxs(
                idxs_to_delete_set, desc="Deleting unvalidated Experiments"
            )
            self._exp_idxs_to_delete_set = None

    def _kill_trial_idxs(self, trial_idxs: Set[int], desc: str = "") -> None:
        gather_fn_kwargs = (
            {"url": f"{self.master_url}/api/v1/trials/{idx}/kill"} for idx in trial_idxs
        )
        asyncio.run(
            self._gather(gather_fn=self._delete_async, gather_fn_kwargs=gather_fn_kwargs, desc=desc)
        )

    def kill_all_active_trials(self, *, safe_mode: bool = True) -> None:
        """Deletes all Experiments from the specified Projects in the Workspace.  Must be called
        twice to perform the deletion when safe_mode == True."""
        trial_idxs_to_kill_set = set()
        for trial_dict in self.get_all_trials():
            if trial_dict["state"] == "STATE_ACTIVE":
                trial_idxs_to_kill_set.add(trial_dict["id"])
        if safe_mode and trial_idxs_to_kill_set != self._trial_idxs_to_kill_set:
            warnings.warn(
                f"This will kill {len(trial_idxs_to_kill_set)} Trials."
                " Please run a second time to confirm deletion."
            )
            self._trial_idxs_to_kill_set = trial_idxs_to_kill_set
        else:
            print(f"Killing {len(trial_idxs_to_kill_set)} experiments.")
            self._kill_trial_idxs(trial_idxs_to_kill_set, desc="Killing all active Trials")
            self._trial_idxs_to_kill_set = None

    def get_trial_summary(
        self,
        trial_idxs: Union[Sequence[int], Set[int], int],
        metric_names: Union[Sequence[str], Set[str], str],
    ) -> List[Dict[str, Any]]:
        """
        Returns a list summarizing each specified trial and the metrics for each specified
        metric name.
        """
        if isinstance(trial_idxs, int):
            trial_idxs = (trial_idxs,)
        if isinstance(metric_names, str):
            metric_names = (metric_names,)
        gather_fn_kwargs = (
            {
                "url": f"{self.master_url}/api/v1/trials/{idx}/summarize",
                "params": {"trialId": idx, "metricNames": metric_names},
            }
            for idx in trial_idxs
        )
        summary = asyncio.run(
            self._gather(
                gather_fn=self._get_json_async,
                gather_fn_kwargs=gather_fn_kwargs,
                desc=f"Getting time series results",
            )
        )
        return summary

    def get_trial_time_series_metrics(
        self,
        trial_idxs: Union[Sequence[int], Set[int], int],
        metric_names: Union[Sequence[str], Set[str], str],
    ) -> Dict[str, Any]:
        """
        Return a dict with the time series for all listed metrics of all listed trials.
        """
        summary = self.get_trial_summary(trial_idxs, metric_names)
        # Index by trial id.
        metrics = {s["trial"]["id"]: s["metrics"] for s in summary}
        return metrics

    def get_trial_time_series_metrics_df(
        self,
        trial_idxs: Union[Sequence[int], Set[int], int],
        metric_names: Union[Sequence[str], Set[str], str],
    ) -> List[Dict[str, Any]]:
        """
        Return a dict with the time series for all listed metrics of all listed trials.
        """
        metrics_dict = self.get_trial_time_series_metrics(trial_idxs, metric_names)
        df_list = []
        for trial_id, metrics in metrics_dict.items():
            for m in metrics:
                name = m["name"]
                step = (d["batches"] for d in m["data"])
                values = (d["value"] for d in m["data"])
                trial_ids = (trial_id for _ in range(len(m["data"])))
                df_list.append(pd.DataFrame({"step": step, name: values, "trial_id": trial_ids}))
        metrics_df = pd.concat(df_list)
        return metrics_df

    async def _gather(
        self,
        gather_fn: Callable,
        gather_fn_kwargs: Generator[Dict[str, Any], None, None],
        desc: str = "",
        timeout_total: int = 600,
    ) -> List[Dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(total=timeout_total)
        async with aiohttp.ClientSession(headers=self._headers, timeout=timeout) as session:
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
