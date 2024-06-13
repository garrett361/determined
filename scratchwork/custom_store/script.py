import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

RANK = os.getenv("RANK")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use gloo, rather than NCCL")

    args = parser.parse_args()
    return args


class PrintTCPStore(dist.TCPStore):
    """
    A TCPStore which just prints all of its actions out
    """

    def set(self, *args, **kwargs) -> None:
        print(f"[{RANK=}]: Calling set with {args=}, {kwargs=}", flush=True)
        super().set(*args, **kwargs)

    def get(self, *args, **kwargs) -> None:
        print(f"[{RANK=}]: Calling get with {args=}, {kwargs=}", flush=True)
        super().set(*args, **kwargs)

    def add(self, *args, **kwargs) -> None:
        print(f"[{RANK=}]: Calling add with {args=}, {kwargs=}", flush=True)
        super().set(*args, **kwargs)

    def compare_set(self, *args, **kwargs) -> None:
        print(f"[{RANK=}]: Calling compare_set with {args=}, {kwargs=}", flush=True)
        super().set(*args, **kwargs)

    def delete_key(self, *args, **kwargs) -> None:
        print(f"[{RANK=}]: Calling delete_key with {args=}, {kwargs=}", flush=True)
        super().set(*args, **kwargs)

    def set_timeout(self, *args, **kwargs) -> None:
        print(f"[{RANK=}]: Calling set_timeout with {args=}, {kwargs=}", flush=True)
        super().set(*args, **kwargs)

    def num_keys(self, *args, **kwargs) -> None:
        print(f"[{RANK=}]: Calling num_keys with {args=}, {kwargs=}", flush=True)
        super().set(*args, **kwargs)

    def wait(self, *args, **kwargs) -> None:
        print(f"[{RANK=}]: Calling wait with {args=}, {kwargs=}", flush=True)
        super().set(*args, **kwargs)


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_port = int(os.environ["MASTER_PORT"])
    master_addr = os.environ["MASTER_ADDR"]

    # Optionally set USE_CPU to a non-trivial value to test using CPU and gloo. Useful for testing
    # independent of OneCCL.
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    store = PrintTCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=world_size,
        is_master=rank == 0,
        multi_tenant=True,
        timeout=timedelta(seconds=60),
        wait_for_workers=True,
    )

    args = get_args()
    backend = "gloo" if args.cpu else "nccl"
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank, store=store)
    dist.barrier()
