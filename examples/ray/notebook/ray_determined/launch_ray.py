import os
import subprocess
import sys
import time
from typing import List

import determined as det


RAY_PORT = 6379
RAY_CLIENT_SERVER_PORT = 10001


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


def create_ngrok_cmd(port: int) -> List[str]:
    return ["ngrok", "tcp", str(port), "--log", '"stdout"']


def main() -> int:
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
        print(f"Launching Ray at {ray_address}")
        ray_proc = subprocess.Popen(ray_cmd)
        time.sleep(5)
        print("Configuring ngrok")
        subprocess.Popen("ngrok config add-authtoken $NGROK_AUTH_TOKEN", shell=True).wait()
        print("Launching ngrok")
        subprocess.Popen(" ".join(launch_cmd), shell=True).wait()
        return ray_proc.wait()


if __name__ == "__main__":
    sys.exit(main())
