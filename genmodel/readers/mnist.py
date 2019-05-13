import torch
from torch.utils import data
from torchvision import datasets, transforms

from pylego.reader import DatasetReader


class MNISTReader(DatasetReader):

    def __init__(self, data_path):
        to_tensor = transforms.ToTensor()
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=to_tensor)
        test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=to_tensor)

        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        torch.manual_seed(0)
        train_dataset, val_dataset = data.random_split(train_dataset, [train_size, val_size])
        super().__init__({'train': train_dataset, 'val': val_dataset, 'test': test_dataset})
