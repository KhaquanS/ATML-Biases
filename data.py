import os
import numpy as np
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

# Custom transform which injects local noise
class AddNoiseToPatch:
    def __init__(self, noise_level=0.1, patch_coords=(0, 0, 50, 50)):
        self.noise_level = noise_level
        self.patch_coords = patch_coords  # (x1, y1, x2, y2)

    def __call__(self, img):
        # Convert to numpy array
        img_np = np.array(img)

        # Extract patch coordinates
        x1, y1, x2, y2 = self.patch_coords
        
        # Generate random noise
        noise = np.random.normal(0, self.noise_level, img_np[y1:y2, x1:x2].shape).astype(np.uint8)

        # Add noise to the patch
        img_np[y1:y2, x1:x2] = np.clip(img_np[y1:y2, x1:x2] + noise, 0, 255)

        # Convert back to PIL Image
        return Image.fromarray(img_np)

class PatchScrambler:
    def __init__(self, patch_size=16):
        self.patch_size = patch_size
    
    def scramble(self, image):
        c, h, w = image.shape
        
        # Checl if image is divisible by patch_size
        assert h % self.patch_size == 0 and w % self.patch_size == 0, "Image size must be divisible by patch size"
        
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        
        # Split image into patches
        patches = image.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        
        # Reshape into (num_patches_h * num_patches_w, C, patch_size, patch_size)
        patches = patches.contiguous().view(c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1, 0, 2, 3)

        # Shuffle the patches
        permuted_indices = torch.randperm(patches.size(0))
        scrambled_patches = patches[permuted_indices]
        
        # Reshape back into original image form
        scrambled_image = scrambled_patches.permute(1, 0, 2, 3).contiguous().view(c, num_patches_h, num_patches_w, self.patch_size, self.patch_size)
        
        # Reassemble the image from scrambled patches
        scrambled_image = scrambled_image.permute(0, 1, 3, 2, 4).contiguous().view(c, h, w)
        
        return scrambled_image
    
    def __call__(self, image):
        return self.scramble(image)


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

def get_noised_data(name, noise_size, root, model_name: Optional[str]=None):

    NOISE_TEST_TFMS = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        AddNoiseToPatch(noise_level=25, patch_coords=(50, 50, 50+noise_size, 50+noise_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if model_name is not None:
        NOISE_TEST_TFMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        AddNoiseToPatch(noise_level=25, patch_coords=(50, 50, 50+noise_size, 50+noise_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],     # Same normalization as CLIP processor
                         std=[0.26862954, 0.26130258, 0.27577711])
    ])

    if name == 'MNIST':
        trainset = torchvision.datasets.MNIST(
            root+'/train', train=True, download=True, transform=TRAIN_TFMS
        )

        normal_testset = torchvision.datasets.MNIST(
            root+'/val', train=False, download=True, transform=TEST_TFMS
        )

        noised_testset = torchvision.datasets.MNIST(
            root+'/noise-val', train=False, download=True, transform=NOISE_TEST_TFMS
        )
        
        labels = torch.cat([clip.tokenize(f"A photo of a number {i}" for i in range(10))])
                                                                                    
    elif name == 'CIFAR-10':
        trainset = torchvision.datasets.CIFAR10(
            root+'/train', train=True, download=True, transform=TRAIN_TFMS
        )

        normal_testset = torchvision.datasets.CIFAR10(
            root+'/val', train=False, download=True, transform=TEST_TFMS
        )

        noised_testset = torchvision.datasets.CIFAR10(
            root+'/noise-val', train=False, download=True, transform=NOISE_TEST_TFMS
        )
        
        labels = torch.cat([clip.tokenize(f"A photo of a {i}" for i in normal_testset.classes)])
                                                                           
    elif name == 'CIFAR-100':
        trainset = torchvision.datasets.CIFAR100(
            root+'/train', train=True, download=True, transform=TRAIN_TFMS
        )

        normal_testset = torchvision.datasets.CIFAR100(
            root+'/val', train=False, download=True, transform=TEST_TFMS
        )

        noised_testset = torchvision.datasets.CIFAR100(
            root+'/noise-val', train=False, download=True, transform=NOISE_TEST_TFMS
        )
                                                                                    
        labels = torch.cat([clip.tokenize(f"A photo of a {i}" for i in normal_testset.classes)])

    
    else:
        raise ValueError('Incorrect dataset name. Choose from [MNIST, CIFAR-10, CIFAR-100].')
    
    if model_name is not None:
        return normal_testset, noised_testset, labels
    
    return trainset, normal_testset, noised_testset

def get_scrambled_data(name, patch_size, root, model_name: Optional[str]=None):
    
    SCRAMBLE_TFMS = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        PatchScrambler(patch_size=patch_size),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if model_name is not None:
        SCRAMBLE_TFMS = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                PatchScrambler(patch_size=patch_size),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],     # Same normalization as CLIP processor
                         std=[0.26862954, 0.26130258, 0.27577711])
            ])
    
    if name.upper() == 'MNIST':
        trainset = torchvision.datasets.MNIST(
            root+'/train', train=True, download=True, transform=TRAIN_TFMS
        )

        normal_testset = torchvision.datasets.MNIST(
            root+'/val', train=False, download=True, transform=TEST_TFMS
        )

        scrambled_testset = torchvision.datasets.MNIST(
            root+'/scrambled-val', train=False, download=True, transform=SCRAMBLE_TFMS
        )
        
        labels = torch.cat([clip.tokenize(f"A photo of a number {digit}" for digit in range(10))])

    elif name.upper() == 'CIFAR-10':
        trainset = torchvision.datasets.CIFAR10(
            root+'/train', train=True, download=True, transform=TRAIN_TFMS
        )

        normal_testset = torchvision.datasets.CIFAR10(
            root+'/val', train=False, download=True, transform=TEST_TFMS
        )

        scrambled_testset = torchvision.datasets.CIFAR10(
            root+'/scrambled-val', train=False, download=True, transform=SCRAMBLE_TFMS
        )    
        
        labels = torch.cat([clip.tokenize(f"A photo of a {i}" for i in normal_testset.classes)])

    elif name == 'CIFAR-100':
        trainset = torchvision.datasets.CIFAR100(
            root+'/train', train=True, download=True, transform=TRAIN_TFMS
        )

        normal_testset = torchvision.datasets.CIFAR100(
            root+'/val', train=False, download=True, transform=TEST_TFMS
        )

        scrambled_testset = torchvision.datasets.CIFAR100(
            root+'/scrambled-val', train=False, download=True, transform=SCRAMBLE_TFMS
        )
        labels = torch.cat([clip.tokenize(f"A photo of a {i}" for i in normal_testset.classes)])

    
    else:
        raise ValueError('Incorrect dataset name. Choose from [MNIST, CIFAR-10, CIFAR-100].')
    
    if model_name is not None:
        return normal_testset, scrambled_testset, labels
    return trainset, normal_testset, scrambled_testset
    

def get_custom_data(train_path: str, val_path: str, model_name: Optional[str] = None, processor: Optional[nn.Module] = None):
    if processor is not None:
        nonstl_dataset = datasets.ImageFolder(train_path, transform=processor)
        stl_dataset = datasets.ImageFolder(val_path, transform=processor)
        labels = torch.cat([clip.tokenize(f"A photo of a {i}" for i in stl_dataset.classes)])

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