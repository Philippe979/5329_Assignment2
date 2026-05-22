import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _build_transforms(use_augmentation: bool = True):
    """
    Build train/test transforms for CIFAR-10.
    """
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, test_transform


def get_cifar10_datasets(
    data_dir: str = "./Data",
    val_ratio: float = 0.1,
    use_augmentation: bool = True,
    seed: int = 42,
):
    """
    Download and prepare CIFAR-10 datasets.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    os.makedirs(data_dir, exist_ok=True)

    train_transform, test_transform = _build_transforms(use_augmentation)

    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # Validation should not use augmentation
    val_dataset.dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=test_transform,
    )

    return train_dataset, val_dataset, test_dataset


def get_cifar10_loaders(
    data_dir: str = "./Data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    use_augmentation: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return train/val/test dataloaders for CIFAR-10.
    """
    train_dataset, val_dataset, test_dataset = get_cifar10_datasets(
        data_dir=data_dir,
        val_ratio=val_ratio,
        use_augmentation=use_augmentation,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader