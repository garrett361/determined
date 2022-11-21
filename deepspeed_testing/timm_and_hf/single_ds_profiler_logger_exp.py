import logging

import determined as det

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    hparams = info.trial.hparams
    with det.core.init() as core_context:
        core_context.train.report_validation_metrics(steps_completed=0, metrics=hparams["results"])
