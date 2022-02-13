import os

import torch
from torch.utils.data import DataLoader, random_split

from torchvision import datasets
from torchvision import transforms

import numpy as np
import random

DATA_DIR = "./data"


def data_path_exists(path):
    return os.path.isdir(f"{DATA_DIR}/{path}")


def load_image_folder(
    folder_name,
    batch_size,
    truncate_number=None,
    val_frac=0.2,
    test_frac=0.1,
    num_workers=0,
):
    """
    Create the data loaders for the CELEBA dataset.

    Parameters:
        folder_name (string): name of the subfolder within the `data` directory
        batch_size (int): the batch size to use
        truncate_number (int): the size of the dataset to truncate to
        val_frac (float): fraction of dataset to be kept for validation set
        test_frac (float): fraction of dataset to be kept for test set
        num_workers (int): number of dataloader workers. NB: increasing this above one will break determinism irrespective of -d flag

    Returns:
        tuple of (training_loader, validation_loader, test_loader)
    """
    if not data_path_exists(folder_name):
        raise IOError(f"No such directory: {DATA_DIR}/{folder_name}")

    if not 0 < val_frac < 1:
        raise ValueError(f"Invalid val_frac. Expected 0-1. Got {val_frac}")
    if not 0 < test_frac < 1:
        raise ValueError(f"Invalid test_frac. Expected 0-1. Got {test_frac}")
    if not 0 < val_frac + test_frac < 1:
        raise ValueError(
            f"Invalid val_frac + test_frac. Expected 0-1. Got {val_frac + test_frac}"
        )

    # Image pre-processing
    trans = [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        # Add transforms here
    ]

    img_dataset = datasets.ImageFolder(
        f"./data/{folder_name}",
        transform=transforms.Compose(trans),
    )

    # Truncate length of dataset
    if truncate_number is not None:
        train_num = min(truncate_number, len(img_dataset))

        # Random truncation of dataset. Seed ensures this is repeatable
        img_dataset, _ = random_split(
            img_dataset,
            [train_num, len(img_dataset) - train_num],
            generator=torch.Generator().manual_seed(42),
        )

    n_val = int(len(img_dataset) * 0.2)
    n_test = int(len(img_dataset) * 0.1)
    n_train = len(img_dataset) - n_val - n_test
    training_set, validation_set, test_set = random_split(
        img_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Tune num_workers to achieve the best GPU utilisation
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    training_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(42),
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers // 4,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(42),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers // 4,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(42),
    )

    return training_loader, validation_loader, test_loader
