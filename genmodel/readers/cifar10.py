import torch
from torch.utils import data
from torchvision import datasets, transforms

from pylego.reader import DatasetReader


class CIFAR10Reader(DatasetReader):

    def __init__(self, data_path):
        to_tensor = transforms.ToTensor()
        train_dataset = datasets.CIFAR10(data_path + '/CIFAR10', train=True, download=True, transform=to_tensor)
        test_dataset = datasets.CIFAR10(data_path + '/CIFAR10', train=False, download=True, transform=to_tensor)
        torch.manual_seed(0)
        super().__init__({'train': data.ConcatDataset([train_dataset, test_dataset])})
