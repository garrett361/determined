import atexit
import json
import os
from pathlib import Path
import requests
import subprocess
import sys
import time
from typing import Dict, List
from urllib import parse

import determined as det
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


def create_launch_cmd_head(proc_per_node: int) -> List[str]:
    cmd = [
        "ray",
        "start",
        "--head",
        "--port",
        str(RAY_PORT),
        "--num-gpus",
        str(proc_per_node),
        "--block",
    ]

    return cmd


def create_launch_cmd_compute(proc_per_node: int, master_addr: str) -> List[str]:
    cmd = [
        "ray",
        "start",
        "--address",
        f"{master_addr}:{RAY_PORT}",
        "--num-gpus",
        str(proc_per_node),
        "--block",
    ]

    return cmd


def create_log_redirect_cmd() -> List[str]:
    return [
        "python3",
        "-m",
        "determined.launch.wrap_rank",
        "RANK",
        "--",
    ]


def create_pid_server_cmd(allocation_id: str, num_slot_ids: int) -> List[str]:
    return [
        "python3",
        "-m",
        "determined.exec.pid_server",
        "--on-fail",
        "SIGTERM",
        "--on-exit",
        "WAIT",
        f"/tmp/pid_server-{allocation_id}",
        str(num_slot_ids),
        "--",
    ]


def create_pid_client_cmd(allocation_id: str) -> List[str]:
    return [
        "python3",
        "-m",
        "determined.exec.pid_client",
        f"/tmp/pid_server-{allocation_id}",
        "--",
    ]


def install_ngrok() -> None:
    subprocess.Popen(
        "curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | "
        "tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && "
        'echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | '
        "tee /etc/apt/sources.list.d/ngrok.list && "
        "apt update && apt install ngrok",
        shell=True,
    ).wait()
    subprocess.Popen("ngrok config add-authtoken $NGROK_AUTH_TOKEN", shell=True).wait()


def create_ngrok_cmd(port: int) -> List[str]:
    return ["ngrok", "tcp", str(port), "--log", '"stdout"']


def main() -> int:
    # install_ngrok()
    info = det.get_cluster_info()
    assert info is not None, "must be run on-cluster"

    os.environ["USE_RAY"] = "True"

    chief_ip = info.container_addrs[0]
    os.environ["DET_CHIEF_IP"] = chief_ip
    if len(info.container_addrs) > 1:
        ray_address = f"ray://{chief_ip}:{RAY_CLIENT_SERVER_PORT}"
    else:
        ray_address = f"ray://localhost:{RAY_CLIENT_SERVER_PORT}"
    os.environ["RAY_ADDRESS"] = ray_address

    if info.container_rank > 0:
        ray_cmd = create_launch_cmd_compute(len(info.slot_ids), chief_ip)
    else:
        ray_cmd = create_launch_cmd_head(len(info.slot_ids))

    log_redirect_cmd = create_log_redirect_cmd()

    pid_server_cmd = create_pid_server_cmd(info.allocation_id, 1)
    pid_client_cmd = create_pid_client_cmd(info.allocation_id)
    ngrok_cmd = create_ngrok_cmd(RAY_CLIENT_SERVER_PORT)

    launch_cmd = pid_server_cmd + pid_client_cmd + log_redirect_cmd + ngrok_cmd

    print(f"Ray launching with: {launch_cmd}")
    print(f"ray cmd {ray_cmd}")
    if info.container_rank > 0:
        print("waiting for chief node")
        time.sleep(5)
        subprocess.Popen(ray_cmd).wait()
        return 0
    else:
        os.environ["RANK"] = "0"
        print("Launching Ray at {ray_address}")
        ray_proc = subprocess.Popen(ray_cmd)
        time.sleep(5)
        print("Configuring ngrok")
        subprocess.Popen("ngrok config add-authtoken $NGROK_AUTH_TOKEN", shell=True).wait()
        print("Launching ngrok")
        subprocess.Popen(" ".join(launch_cmd), shell=True).wait()
        return ray_proc.wait()


def init_ray_determined(
    num_slots: int = 1,
    ngrok_auth_token: str = "",
    det_master: str = "http://127.0.0.1:8080",
    det_user: str = "determined",
    det_password: str = "",
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
        "resources": {"slots_per_trial": num_slots},
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
                url = log[idx + 4 :]
                # Replace tcp:// with ray://
                url = "ray" + url[3:]
                found = True
                break
        if found:
            break
        time.sleep(1)
    ray.init(url)

    def kill_exp() -> None:
        exp.kill()

    # This isn't fully reliable; cluster may need to be manually killed sometimes.
    # Might be nice to add a wrapper in the task that checks continued existence of initiating
    # process via ping.
    atexit.register(kill_exp)
    print("Ray initialized.")


if __name__ == "__main__":
    sys.exit(main())
