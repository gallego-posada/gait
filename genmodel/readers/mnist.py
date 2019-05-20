import torch
from torch.utils import data
from torchvision import datasets, transforms

from pylego.reader import DatasetReader


class MNISTReader(DatasetReader):

    def __init__(self, data_path):
        to_tensor = transforms.ToTensor()
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=to_tensor)
        test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=to_tensor)
        torch.manual_seed(0)
        super().__init__({'train': data.ConcatDataset([train_dataset, test_dataset])})
