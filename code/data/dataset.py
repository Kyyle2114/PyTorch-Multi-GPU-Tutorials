import torchvision
from torchvision.models import ResNet50_Weights

import torch

from typing import Tuple

def download_CIFAR10(root='cifar10') -> None:
    """
    Download CIFAR10 dataset
    Location : code/cifar10/

    Args:
        root (str, optional): path to save. Defaults to 'cifar10'.
    """
    torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    torchvision.datasets.CIFAR10(root=root, train=False, download=True)

def load_CIFAR10(root='cifar10') -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load CIFAR10 dataset
    Apply ToTensor(), Normalize() transforms.

    Args:
        root (str, optional): path to load. Defaults to 'cifar10'.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: pytorch dataset
    """
    
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
    
    return train_set, test_set
        