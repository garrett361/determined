import atexit
import json
import os
from pathlib import Path
import requests
import time
from typing import Any, Dict, List, Optional
from urllib import parse

from determined.experimental import client
import ray


RAY_PORT = 6379
RAY_CLIENT_SERVER_PORT = 10001


def parse_master_address(master_address: str) -> parse.ParseResult:
    if master_address.startswith("https://"):
        default_port = 443
    elif master_address.startswith("http://"):
        default_port = 80
    else:
        default_port = 8080
        master_address = "http://{}".format(master_address)
    parsed = parse.urlparse(master_address)
    if not parsed.port:
        parsed = parsed._replace(netloc="{}:{}".format(parsed.netloc, default_port))
    return parsed


def make_url(master_address: str, suffix: str) -> str:
    parsed = parse_master_address(master_address)
    return parse.urljoin(parsed.geturl(), suffix)


class AuthException(Exception):
    pass


class DeterminedMasterAPI:
    """
    Maintains authorization state and provides Determined REST API calls.
    """

    def __init__(self, master_url: str) -> None:
        self.auth_token = None
        self.master_url = master_url

    def _get_auth_header(self) -> Dict[str, str]:
        if self.auth_token is None:
            raise AuthException(
                "Must retrieve token via `login` before making calls that require auth."
            )
        return {"Authorization": f"Bearer {self.auth_token}"}

    def login(self, username: str, password: str) -> None:
        url = make_url(self.master_url, "/api/v1/auth/login")
        resp = requests.post(url, json={"username": username, "password": password})
        self.auth_token = resp.json()["token"]

    def get_logs(self, trial_id: int) -> List[str]:
        """
        Get logs for the given trial ID.
        """
        url = make_url(self.master_url, f"/api/v1/trials/{trial_id}/logs")
        params = {
            "limit": "0",
            "follow": "false",
            "orderBy": "ORDER_BY_ASC",
        }
        resp = requests.get(url, headers=self._get_auth_header(), params=params)
        ret = []
        json_strs = resp.text.split("\n")
        for s in json_strs:
            s = s.strip()
            if not (s):
                continue
            ret.append(json.loads(s)["result"]["message"])
        return ret


def init_ray_determined(
    num_slots: int = 1,
    ngrok_auth_token: str = "",
    det_master: str = "http://127.0.0.1:8080",
    det_user: str = "determined",
    det_password: str = "",
    ray_init_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    ngrok_auth_token = ngrok_auth_token or os.environ["NGROK_AUTH_TOKEN"]
    if det_master is None:
        det_master = os.environ["DET_MASTER"]
    if det_user is None:
        det_user = os.environ["DET_USER"]
    if det_password is None:
        det_password = os.environ["DET_PASSWORD"]
    client.login(master=det_master, user=det_user, password=det_password)
    config = {
        "name": "ray cluster",
        "debug": False,
        # 10GB shm_size. Ray gives warning for default size.
        "resources": {"slots_per_trial": num_slots, "shm_size": 10737418240},
        "searcher": {"name": "single", "metric": "loss", "max_length": {"batches": 1}},
        "entrypoint": "python3 launch_ray.py",
        "max_restarts": 0,
        "environment": {
            "image": {"cuda": "coreystaten/raynotebook:latest"},
            "environment_variables": [f"NGROK_AUTH_TOKEN={ngrok_auth_token}"],
        },
    }
    this_file_dir = Path(__file__).resolve().parent
    exp = client.create_experiment(config, model_dir=this_file_dir)
    print("Waiting for trial ID")
    while True:
        trials = exp.get_trials()
        if trials:
            trial_id = trials[0].id
            break
        time.sleep(1)
    master_api = DeterminedMasterAPI(det_master)
    master_api.login(det_user, det_password)
    print("Waiting for Ray master to initialize...")
    found = False
    while True:
        # Wait for ngrok
        logs = master_api.get_logs(trial_id)
        for log in logs:
            if "started tunnel" in log:
                idx = log.find("url=")
                tcp_url = log[idx + 4 :]
                # Replace tcp:// with ray://
                ray_url = "ray" + tcp_url[3:]
                found = True
                break
        if found:
            break
        time.sleep(1)
    ray_init_kwargs = ray_init_kwargs or {}
    ray.init(address=ray_url, **ray_init_kwargs)
    print(f"Ray initialized at address {ray_url}")

    def kill_exp() -> None:
        exp.kill()

    # This isn't fully reliable; cluster may need to be manually killed sometimes.
    # Might be nice to add a wrapper in the task that checks continued existence of initiating
    # process via ping.
    atexit.register(kill_exp)
    print("Ray initialized.")
