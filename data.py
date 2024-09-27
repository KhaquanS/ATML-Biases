import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision import datasets
import deeplake
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import clip

IMAGE_SIZE = 224
TRAIN_TFMS = transforms.Compose([
    transforms.RandAugment(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
TEST_TFMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def get_custom_data(train_path: str, val_path: str, model_name: Optional[str] = None, processor: Optional[nn.Module] = None):
    if processor is not None:
        nonstl_dataset = datasets.ImageFolder(train_path, transform=processor)
        stl_dataset = datasets.ImageFolder(val_path, transform=processor)
        labels = torch.cat([clip.tokenize(f"A photo of a {i}" for i in val_dataset.classes)])

        return nonstl_dataset, stl_dataset, labels

    train_dataset = datasets.ImageFolder(train_path, transform=TRAIN_TFMS)
    val_dataset = datasets.ImageFolder(val_path, transform=TEST_TFMS)

    return train_dataset, val_dataset

def get_dataset_func(name, model_name: Optional[str] = None, processor: Optional[nn.Module] = None, out_dir: Optional[str] = None):
    if model_name.startswith("clip"):
        if name == 'MNIST':
            return get_MNIST_dataset(root=f'./{out_dir}/{name}', processor=processor)
        elif name == 'CIFAR-10':
            return get_CIFAR10_dataset(root=f'./{out_dir}/{name}',processor=processor)
        elif name == 'CIFAR-100':
            return get_CIFAR100_dataset(root=f'./{out_dir}/{name}',processor=processor)
        elif name == 'PACS':
            return get_PACS_dataset(root=f'./{out_dir}/{name}', processor=processor)
        elif name == 'SVHN':
            return get_SVHN_dataset(root=f'./{out_dir}/{name}',processor=processor)
        else:
            raise ValueError("Received invalid dataset name - please check data.py")
        
    if name == 'MNIST':
        return get_MNIST_dataset
    elif name == 'CIFAR-10':
        return get_CIFAR10_dataset
    elif name == 'CIFAR-100':
        return get_CIFAR100_dataset
    elif name == 'PACS':
        return get_PACS_dataset
    elif name == 'SVHN':
        return get_SVHN_dataset
    else:
        raise ValueError("Received invalid dataset name - please check data.py")
    
def get_dataloader(dataset: Dataset,
                   batch_size: int,
                   is_train: bool,
                   num_workers: int = 1):
    
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=is_train, num_workers=num_workers)
    return loader

def get_deeplake_dataloader(
        train_data,
        test_data,
        batch_size: int,
        num_workers: int=1,
        processor: Optional[nn.Module]=None):
    
    if processor is not None:
        train_loader = train_data.pytorch(num_workers = num_workers, shuffle = True, transform = {'images': processor, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
        test_loader = test_data.pytorch(num_workers = num_workers, transform = {'images': processor, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
                
        return train_loader, test_loader

    train_loader = train_data.pytorch(num_workers = num_workers, shuffle = True, transform = {'images': TRAIN_TFMS, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})
    test_loader = test_data.pytorch(num_workers = num_workers, transform = {'images': TEST_TFMS, 'labels': None}, batch_size = batch_size, decode_method = {'images': 'pil'})

    return train_loader, test_loader

def get_MNIST_dataset(root: str, processor: Optional[nn.Module]=None):

    if processor is not None:
        trainset = torchvision.datasets.MNIST(
        root, train=True, download=True, transform=processor)

        testset = torchvision.datasets.MNIST(
        root, train=False, download=True, transform=processor)

        labels = torch.cat([clip.tokenize(f"A photo of a number {digit}" for digit in range(10))])

        return trainset, testset, labels


    trainset = torchvision.datasets.MNIST(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.MNIST(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset

def get_CIFAR10_dataset(root: str, processor: Optional[nn.Module]=None):

    if processor is not None:
        trainset = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=processor)

        testset = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=processor)
        labels = torch.cat([clip.tokenize(f"A photo of a {i}" for i in testset.classes)])

        return trainset, testset, labels


    trainset = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset

def get_CIFAR100_dataset(root: str, processor: Optional[nn.Module]=None):

    if processor is not None:
        trainset = torchvision.datasets.MNIST(
        root, train=True, download=True, transform=processor)

        testset = torchvision.datasets.MNIST(
        root, train=False, download=True, transform=processor)
        labels = torch.cat([clip.tokenize(f"A photo of a {i}" for i in testset.classes)])

        return trainset, testset, labels

    trainset = torchvision.datasets.CIFAR100(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.CIFAR100(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset

def get_PACS_dataset(root: str, processor: Optional[nn.Module]=None):

    trainset = deeplake.load('hub://activeloop/pacs-train')
    testset = deeplake.load('hub://activeloop/pacs-val')
    
    if processor is not None:
        labels = torch.cat([clip.tokenize(f"A photo of a {i}" for i in trainset.labels.info["class_names"])])
        return trainset, testset, labels

    return trainset, testset

def get_SVHN_dataset(root: str, processor: Optional[nn.Module]=None):

    if processor is not None:
        trainset = torchvision.datasets.SVHN(
        root, split="train", download=True, transform=processor)

        testset = torchvision.datasets.SVHN(
        root, split="test", download=True, transform=processor)
        labels = torch.cat([clip.tokenize(f"A photo of a digit {i}" for i in range(10))])

        return trainset, testset, labels

    trainset = torchvision.datasets.SVHN(
        root, split='train', download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.SVHN(
        root, split='test', download=True, transform=TEST_TFMS
    )

    return trainset, testset
