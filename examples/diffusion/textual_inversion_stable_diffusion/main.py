"""Following the SD textual inversion notebook example from HF
https://github.com/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb"""

import determined as det
import logging

from detsd import DetStableDiffusionTITrainer

logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)


if __name__ == "__main__":
    trainer = DetStableDiffusionTITrainer.init_on_cluster()
    trainer.train()
