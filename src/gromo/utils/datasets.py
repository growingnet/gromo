from math import ceil, floor
from warnings import warn

import torch
from torch.utils import data
from torchvision import datasets, transforms


def get_dataset(
    dataset_name: str,
    dataset_path: str,
    nb_class: int | None = None,
    split_train_val: float = 0.1,
    data_augmentation: list[str] | None = None,
    seed: int | None = 0,
    **kwargs,
) -> tuple[data.Dataset, data.Dataset, data.Dataset]:
    """
    Get the dataset

    Parameters
    ----------
    dataset_name: str
        The name of the dataset
    dataset_path: str
        The path to the dataset or where to download it
    nb_class: int | None
        The number of classes to keep in the dataset, for example, if nb_class=5, only the first 5 classes will be kept
    split_train_val: float
        The proportion of the training set to use as validation set
    data_augmentation: list[str] | None
        The data augmentation to apply to the training set
    seed: int | None
        The seed to use for the random number generator
    kwargs: dict
        Additional arguments

    Returns
    -------
    tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]
        The training, validation, and test datasets
    """
    known_datasets = {
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }
    if dataset_name not in known_datasets:
        raise ValueError(f"Unknown dataset {dataset_name}")

    mnist_mean = torch.tensor(0.1307)
    mnist_std = torch.tensor(0.3081)

    datasets_transforms = {
        "mnist": [
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std),
            transforms.Lambda(lambda x: x.view(-1)),
        ],
        "cifar10": [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
            ),
            transforms.Lambda(lambda x: x.view(-1)),
        ],
    }
    augmentation_transforms = []
    if data_augmentation:
        for aug in data_augmentation:
            if aug == "horizontal_flip":
                augmentation_transforms.append(transforms.RandomHorizontalFlip())
            elif aug == "rotation":
                augmentation_transforms.append(transforms.RandomRotation(10))
            elif aug == "crop":
                augmentation_transforms.append(
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect")
                )
            else:
                warn(f"Unknown augmentation {aug}")
            # Add more augmentations as needed

    dataset = known_datasets[dataset_name]
    datasets_transform = datasets_transforms[dataset_name]

    train_val_data = dataset(
        root=dataset_path,
        train=True,
        download=True,
        transform=transforms.Compose(augmentation_transforms + datasets_transform),
    )

    test_data = dataset(
        root=dataset_path,
        train=False,
        download=True,
        transform=transforms.Compose(datasets_transform),
    )

    if nb_class is not None and nb_class <= 1:
        warn(f"{nb_class=} should be greater than 1")
    if nb_class is not None:
        targets = train_val_data.targets
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        assert isinstance(targets, torch.Tensor)
        assert targets.ndim == 1
        assert targets.min() == 0
        train_idx = targets <= (nb_class - 1)
        train_val_data.data = train_val_data.data[train_idx]
        train_val_data.targets = targets[train_idx]

        targets = test_data.targets
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        assert isinstance(targets, torch.Tensor)
        assert targets.ndim == 1
        assert targets.min() == 0
        test_idx = targets <= (nb_class - 1)
        test_data.data = test_data.data[test_idx]
        test_data.targets = targets[test_idx]

    if split_train_val < 0 or split_train_val > 1:
        raise ValueError(f"{split_train_val=} should be in [0, 1]")
    if split_train_val > 0.5:
        warn(
            f"{split_train_val=} is greater than 0.5 this means that "
            f"the validation set is greater than the training set"
        )
    train_size = ceil(len(train_val_data) * (1 - split_train_val))
    val_size = floor(len(train_val_data) * split_train_val)
    assert train_size + val_size == len(train_val_data)

    if seed is not None:
        torch.manual_seed(seed)
    train_data, val_data = torch.utils.data.random_split(
        train_val_data, [train_size, val_size]
    )

    return train_data, val_data, test_data
