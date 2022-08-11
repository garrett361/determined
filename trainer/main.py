import torch

import data
import models
from trainer import Trainer


def main() -> None:
    model_class = models.MNISTModel
    optimizer_class = torch.optim.Adam
    train_dataset = data.get_mnist_dataset(train=True)
    val_dataset = data.get_mnist_dataset(train=False)
    trainer = Trainer(
        model_class,
        optimizer_class,
        train_dataset,
        val_dataset,
    )
    trainer.run()


if __name__ == "__main__":
    main()
