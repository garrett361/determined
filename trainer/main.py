import torch

import data
import models
from trainer import Trainer


def main() -> None:
    trainer = Trainer(
        model_class=models.MNISTModel,
        optimizer_class=torch.optim.Adam,
        criterion_class=torch.nn.CrossEntropyLoss,
        train_dataset=data.get_mnist_dataset(train=True),
        val_dataset=data.get_mnist_dataset(train=False),
    )
    trainer.run()


if __name__ == "__main__":
    main()
