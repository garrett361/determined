#!/usr/bin/env python3

import random
import time

from determined.experimental import core_v2


def test_ckpt_storage():
    core_v2.init(
        defaults=core_v2.DefaultConfig(
            name="unmanaged-2-checkpoints",
            # We allow configuring the local checkpoint storage directory.
            checkpoint_storage="/tmp/determined-cp",
        ),
        unmanaged=core_v2.UnmanagedConfig(
            external_experiment_id=f"{int(time.time())} ",
            external_trial_id=f"{int(time.time())} ",
            workspace="garrett",
            project="garrett"
            # e.g., requeued jobs on slurm:
            # external_experiment_id=f"some-prefix-{os.environ[SLURM_JOB_ID}",
            # external_trial_id=f"some-prefix-{os.environ[SLURM_JOB_ID}",
        ),
    )

    latest_checkpoint = core_v2.info.latest_checkpoint
    print("latest_checkpoint", latest_checkpoint)
    initial_i = 0
    if latest_checkpoint is not None:
        with core_v2.checkpoint.restore_path(latest_checkpoint) as path:
            with (path / "state").open() as fin:
                ckpt = fin.read()
                print("Checkpoint contents:", ckpt)

                i_str, _ = ckpt.split(",")
                initial_i = int(i_str)

    print("determined experiment id: ", core_v2.info._trial_info.experiment_id)
    print("initial step:", initial_i)
    for i in range(initial_i, initial_i + 10):
        core_v2.train.report_training_metrics(steps_completed=i, metrics={"loss": random.random()})
        if (i + 1) % 5 == 0:
            loss = random.random()
            core_v2.train.report_validation_metrics(steps_completed=i, metrics={"loss": loss})

            with core_v2.checkpoint.store_path({"steps_completed": i}) as (path, uuid):
                with (path / "state").open("w") as fout:
                    fout.write(f"{i},{loss}")
        if (i + 1) % 6 == 0:
            raise RuntimeError("Erroring to test out #802")

    core_v2.close()


if __name__ == "__main__":
    test_ckpt_storage()
