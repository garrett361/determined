import torchvision.datasets as datasets
import torchvision.transforms as T

mnist_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ]
)


def get_mnist_dataset(train: bool, root: str = "shared_fs/data") -> datasets.MNIST:
    return datasets.MNIST(root=root, train=train, download=True, transform=mnist_transform)
