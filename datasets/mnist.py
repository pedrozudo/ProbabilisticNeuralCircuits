from pathlib import Path

from torchvision.datasets import MNIST as BaseMNIST
from torchvision.datasets import FashionMNIST as BaseFashionMNIST
from torchvision.datasets import EMNIST as BaseEMNIST
import torchvision


MNIST_DATASETS = [
    "mnist",
    "fmnist",
    "emnist:mnist",
    "emnist:letters",
    "emnist:balances",
    "emnist:byclass",
]
N_CLASSES = {"mnist": 10, "fmnist": 10}


DATA_PATH = Path(__file__).parent.resolve()
TRANSFORMS = [torchvision.transforms.ToTensor()]
TRANSFORMS = torchvision.transforms.Compose(TRANSFORMS)


class MNIST(BaseMNIST):
    def __init__(self, mode):
        super().__init__(
            DATA_PATH,
            train=(mode == "train"),
            download=True,
            transform=TRANSFORMS,
        )

    def __getitem__(self, index):
        x = self.data[index, :]
        y = self.targets[index]
        return {"x": x, "y": y}


class FashionMNIST(BaseFashionMNIST):
    def __init__(self, mode):
        super().__init__(
            DATA_PATH,
            train=(mode == "train"),
            download=True,
            transform=TRANSFORMS,
        )

    def __getitem__(self, index):
        x = self.data[index, :]
        y = self.targets[index]
        return {"x": x, "y": y}


class EMNIST(BaseEMNIST):
    def __init__(self, split, mode):
        super().__init__(
            DATA_PATH,
            split,
            train=(mode == "train"),
            download=True,
            transform=TRANSFORMS,
        )

    def __getitem__(self, index):
        x = self.data[index, :]
        y = self.targets[index]
        return {"x": x, "y": y}
