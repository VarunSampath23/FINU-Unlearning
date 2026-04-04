"""
src/datasets.py

Handles dataset loading, transformations, and forget/retain splitting
for Class, Subclass, and Sample unlearning on CIFAR-100, CIFAR Super-20, and ImageNet.
"""

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100, ImageNet
from collections import defaultdict
from typing import Tuple, List, Optional, Dict

# ====================== Custom Dataset for Super-20 ======================
class CifarSuper20(CIFAR100):
    """
    CIFAR-100 with 20 superclasses (coarse labels).
    Returns (image, fine_label, coarse_label)
    """
    def __init__(self, root: str = ".", train: bool = True, download: bool = True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

        self.coarse_map = {
            0: [4, 30, 55, 72, 95],   # aquatic mammals
            1: [1, 32, 67, 73, 91],   # fish
            2: [54, 62, 70, 82, 92],  # flowers
            3: [9, 10, 16, 28, 61],   # food containers
            4: [0, 51, 53, 57, 83],   # fruit and vegetables
            5: [22, 39, 40, 86, 87],  # household electrical devices
            6: [5, 20, 25, 84, 94],   # household furniture
            7: [6, 7, 14, 18, 24],    # insects
            8: [3, 42, 43, 88, 97],   # large carnivores
            9: [12, 17, 37, 68, 76],  # large man-made outdoor things
            10: [23, 33, 49, 60, 71], # large natural outdoor scenes
            11: [15, 19, 21, 31, 38], # large omnivores and herbivores
            12: [34, 63, 64, 66, 75], # medium-sized mammals
            13: [26, 45, 77, 79, 99], # non-insect invertebrates
            14: [2, 11, 35, 46, 98],  # people
            15: [27, 29, 44, 78, 93], # reptiles
            16: [36, 50, 65, 74, 80], # small mammals
            17: [47, 52, 56, 59, 96], # trees
            18: [8, 13, 48, 58, 90],  # vehicles 1
            19: [41, 69, 81, 85, 89]  # vehicles 2
        }

    def __getitem__(self, index: int):
        img, fine_label = super().__getitem__(index)
        coarse_label = None
        for c, fine_list in self.coarse_map.items():
            if fine_label in fine_list:
                coarse_label = c
                break
        assert coarse_label is not None, f"Label {fine_label} not found in coarse_map"
        return img, fine_label, coarse_label


# ====================== Transforms ======================
def get_transforms(dataset_name: str, train: bool = True):
    if dataset_name in ["cifar100", "cifar_super20"]:
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    elif dataset_name == "imagenet":
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ====================== Main Dataloader Function ======================
def get_dataloaders(
    dataset_name: str = "cifar100",
    forget_type: Optional[str] = None,      # None = full dataset (for base training)
    forget_class: Optional[int] = None,
    forget_classes: Optional[List[int]] = None,
    forget_size: Optional[int] = None,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, int]:
    """
    Returns:
        forget_train_dl, retain_train_dl, forget_test_dl, retain_test_dl, test_dl, num_classes
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    transform_train = get_transforms(dataset_name, train=True)
    transform_test = get_transforms(dataset_name, train=False)

    if dataset_name == "cifar100":
        train_ds = CIFAR100(root=".", train=True, download=True, transform=transform_train)
        test_ds = CIFAR100(root=".", train=False, download=True, transform=transform_test)
        num_classes = 100
        label_key = 'targets'

    elif dataset_name == "cifar_super20":
        train_ds = CifarSuper20(root=".", train=True, download=True, transform=transform_train)
        test_ds = CifarSuper20(root=".", train=False, download=True, transform=transform_test)
        num_classes = 20
        label_key = None  # we use coarse label (index 2)

    elif dataset_name == "imagenet":
        train_ds = ImageNet(root="./ImageNet", split="train", transform=transform_train)
        test_ds = ImageNet(root="./ImageNet", split="val", transform=transform_test)
        num_classes = 1000
        label_key = 'targets'
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    # ====================== Full dataset for base training ======================
    if forget_type is None:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
        # Return dummy loaders for compatibility
        return None, train_dl, None, None, test_dl, num_classes

    # ====================== Forget / Retain splitting ======================
    if dataset_name == "cifar_super20":
        # Use coarse label (third returned value)
        def get_label(item):
            return item[2] if isinstance(item, tuple) else item[2]
    else:
        def get_label(item):
            return item[1] if isinstance(item, tuple) else getattr(train_ds, label_key)[item]

    # Build index lists
    if forget_type == "class":
        if forget_class is None:
            raise ValueError("forget_class must be provided for class unlearning")
        forget_classes = [forget_class]

        forget_train_idx = [i for i, lbl in enumerate(getattr(train_ds, label_key, None) or 
                            [get_label(train_ds[i]) for i in range(len(train_ds))]) 
                            if lbl in forget_classes]
        retain_train_idx = [i for i in range(len(train_ds)) if i not in forget_train_idx]

        forget_test_idx = [i for i, lbl in enumerate(getattr(test_ds, label_key, None) or 
                           [get_label(test_ds[i]) for i in range(len(test_ds))]) 
                           if lbl in forget_classes]
        retain_test_idx = [i for i in range(len(test_ds)) if i not in forget_test_idx]

    elif forget_type == "subclass":
        if forget_classes is None:
            raise ValueError("forget_classes list must be provided for subclass unlearning")
        forget_train_idx = []
        for c in forget_classes:
            forget_train_idx.extend([i for i, lbl in enumerate([get_label(train_ds[j]) for j in range(len(train_ds))]) 
                                     if lbl == c])
        retain_train_idx = [i for i in range(len(train_ds)) if i not in set(forget_train_idx)]

        forget_test_idx = []
        for c in forget_classes:
            forget_test_idx.extend([i for i, lbl in enumerate([get_label(test_ds[j]) for j in range(len(test_ds))]) 
                                    if lbl == c])
        retain_test_idx = [i for i in range(len(test_ds)) if i not in set(forget_test_idx)]

    elif forget_type == "sample":
        if forget_size is None:
            raise ValueError("forget_size must be provided for sample unlearning")
        total_indices = list(range(len(train_ds)))
        forget_train_idx = random.sample(total_indices, forget_size)
        retain_train_idx = list(set(total_indices) - set(forget_train_idx))

        # For test, we use full test set as "test_dl"
        forget_test_idx = []
        retain_test_idx = list(range(len(test_ds)))

    else:
        raise ValueError(f"Unknown forget_type: {forget_type}")

    # Create subsets
    forget_train = Subset(train_ds, forget_train_idx)
    retain_train = Subset(train_ds, retain_train_idx)
    forget_test = Subset(test_ds, forget_test_idx)
    retain_test = Subset(test_ds, retain_test_idx)

    # Create dataloaders
    forget_train_dl = DataLoader(forget_train, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=pin_memory)
    retain_train_dl = DataLoader(retain_train, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=pin_memory)
    forget_test_dl = DataLoader(forget_test, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
    retain_test_dl = DataLoader(retain_test, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)

    return forget_train_dl, retain_train_dl, forget_test_dl, retain_test_dl, test_dl, num_classes


# ====================== Utility ======================
def get_full_train_loader(dataset_name: str = "cifar100", batch_size: int = 256, seed: int = 42):
    """Convenience function for base model training."""
    _, full_train_dl, _, _, _, _ = get_dataloaders(
        dataset_name=dataset_name,
        forget_type=None,
        batch_size=batch_size,
        seed=seed
    )
    return full_train_dl
