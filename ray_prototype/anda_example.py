import logging
import os

import determined as det
import ray
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.models as models
from determined.pytorch import DataLoader
from ray.air import session
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms


@ray.remote(num_gpus=8)
class TrainModel:
    def __init__(self, model, model_name, training_dataset, validation_dataset):
        self.model = model
        self.model_name = model_name
        self.device = torch.device("cuda")
        self.train_dataset = training_dataset
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.validation_dataset = validation_dataset

    def train(self, metrics_step: int = 10):
        self.model = self.model.to(self.device)
        results = {}
        for batch_idx, (input, target) in enumerate(self.train_dataset):
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            input = input.to(self.device)
            target = target.to(self.device)

            output = self.model(input)

            loss = self.loss_fn(output, target)
            loss = loss.to(self.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % metrics_step == 0:
                results[batch_idx] = {
                    f"{self.model_name}_loss": loss.item(),
                }
        return results

    def validate(self, metrics_step: int = 10):
        results = {}
        self.model = self.model.to(self.device)
        self.model.eval()
        for batch_idx, (input, target) in enumerate(self.validation_dataset):
            input = input.to(self.device)
            target = target.to(self.device)
            output = self.model(input)

            loss = self.loss_fn(output, target)
            loss = loss.to(self.device)
            pred = output.argmax(dim=1, keepdim=True)
            accuracy = pred.eq(target.view_as(pred)).sum().item() / len(input)

            if batch_idx % metrics_step == 0:
                results[batch_idx] = {
                    f"{self.model_name}_validation_loss": loss.item(),
                    f"{self.model_name}_validation_accuracy": accuracy,
                }
        return results


def get_training_dataset(length: int, batch_size: int):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    training_dataset = datasets.ImageFolder("/data/imagenet/train", transform=transform)
    training_dataset = torch.utils.data.Subset(training_dataset, range(0, length))
    return DataLoader(training_dataset, batch_size=batch_size)


def get_validation_dataset(length: int, batch_size: int):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    training_dataset = datasets.ImageFolder("/data/imagenet/validation", transform=transform)
    training_dataset = torch.utils.data.Subset(training_dataset, range(0, length))
    return DataLoader(training_dataset, batch_size=batch_size)


def merge_dictionary_list(dict_list):
    return {k: [d.get(k) for d in dict_list if k in d] for k in set().union(*dict_list)}


def main():
    train_tasks = []
    base_model = models.efficientnet_b0(pretrained=False)
    ensemble_models = [base_model]
    training_dataset = get_training_dataset(25000, 32)
    validation_dataset = get_validation_dataset(5000, 32)

    ray.init(num_gpus=8)
    ray_actors = []
    for model_idx, model in enumerate(ensemble_models):
        actor = TrainModel.remote(
            model=model,
            model_name=f"efficientnet_{model_idx}",
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
        )
        ray_actors.append(actor)

    for actor in ray_actors:
        train_tasks.append(actor.train.remote())

    training_metrics = merge_dictionary_list(ray.get(train_tasks))

    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    distributed = det.core.DummyDistributedContext()

    with det.core.init(distributed=distributed) as core_context:
        for batch_idx in training_metrics:
            core_context.train.report_training_metrics(
                steps_completed=batch_idx,
                metrics={k: v for d in training_metrics[batch_idx] for k, v in d.items()},
            )

    validation_tasks = []

    for actor in ray_actors:
        validation_tasks.append(actor.validate.remote())

    validation_metrics = merge_dictionary_list(ray.get(validation_tasks))
    with det.core.init(distributed=distributed) as core_context:
        for batch_idx in validation_metrics:
            core_context.train.report_validation_metrics(
                steps_completed=batch_idx,
                metrics={k: v for d in validation_metrics[batch_idx] for k, v in d.items()},
            )


if __name__ == "__main__":
    info = det.get_cluster_info()
    main()
