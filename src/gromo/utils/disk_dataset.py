import os
import warnings
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class DiskDataset(IterableDataset):
    """Dataset that loads data from the disk

    Parameters
    ----------
    input_filename : str
        file name of input data dictionary
    target_filename : str
        filename of target data dictionary
    input_keys : list[str]
        keys to concatenate for the input
    target_keys : list[str]
        keys to concatenate for the target
    """

    def __init__(
        self,
        input_filename: str,
        target_filename: str,
        input_keys: list[str],
        target_keys: list[str],
    ) -> None:
        self.input_filename = input_filename
        self.target_filename = target_filename
        self.input_keys = input_keys
        self.target_keys = target_keys

    def load_data(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Load data from disk and yield features and target

        Yields
        ------
        tuple[torch.Tensor, torch.Tensor]
            x, y

        Raises
        ------
        ValueError
            if target and input data do not have the same length
        """
        input_t = torch.load(self.input_filename)
        target_t = torch.load(self.target_filename)

        input_t = torch.cat([input_t[k] for k in self.input_keys], dim=1)
        target_t = torch.cat([target_t[k] for k in self.target_keys], dim=1)

        if len(input_t) != len(target_t):
            raise ValueError(
                f"Input and target tensors must have same length, "
                f"got {len(input_t)} and {len(target_t)}"
            )

        for x, y in zip(input_t, target_t):
            yield x, y

    def __iter__(self) -> Iterator:
        yield from self.load_data()


class MemMapDataset(Dataset):
    """Dataset for lazy loading from the disk.
    Datapoints are read one by one from the disk.

    Parameters
    ----------
    input_filename : str
        file name of input data dictionary
    target_filename : str
        file name of target data dictionary
    input_keys : list[str]
        keys to concatenate for the input
    target_keys : list[str]
        keys to concatenate for the target

    Raises
    ------
    ValueError
        if target and input data do not have the same length
    """

    def __init__(
        self,
        input_filename: str,
        target_filename: str,
        input_keys: list[str],
        target_keys: list[str],
    ) -> None:
        self.input_filename = input_filename
        self.target_filename = target_filename
        self.input_keys = input_keys
        self.target_keys = target_keys

        input_name = self.__convert_dict_to_npy(
            self.input_filename, self.input_keys, "__input"
        )
        target_name = self.__convert_dict_to_npy(
            self.target_filename, self.target_keys, "__target"
        )

        self.input_map = np.load(input_name, mmap_mode="r")
        self.target_map = np.load(target_name, mmap_mode="r")

        if len(self.input_map) != len(self.target_map):
            raise ValueError(
                f"Input and target tensors must have same length, "
                f"got {len(self.input_map)} and {len(self.target_map)}"
            )

    def __convert_dict_to_npy(self, path: str, keys: list[str], name: str) -> str:
        data = torch.load(path)
        data = torch.cat([data[k] for k in keys], dim=1)
        file_name = os.path.join(os.path.dirname(path), f"{name}.npy")
        np.save(file_name, data.numpy())
        return file_name

    def __len__(self) -> int:
        return len(self.input_map)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                category=UserWarning,
                message="The given NumPy array is not writable",
                action="ignore",
            )
            x = torch.from_numpy(self.input_map[index])
            y = torch.from_numpy(self.target_map[index])
        return x, y


class SimpleMemMapDataset(Dataset):
    """Dataset for lazy loading from the disk.
    Datapoints are read one by one from the disk.
    Assumes that the data keys are in separate files.

    Parameters
    ----------
    input_filenames : list[str]
        file names of input data to concatenate
    target_filenames : list[str]
        file names of target data to concatenate

    Raises
    ------
    ValueError
        if target and input data do not have the same length
    """

    def __init__(self, input_filenames: list[str], target_filenames: list[str]) -> None:
        self.input_filenames = input_filenames
        self.target_filenames = target_filenames

        self.input_maps = [np.load(name, mmap_mode="r") for name in self.input_filenames]
        self.target_maps = [
            np.load(name, mmap_mode="r") for name in self.target_filenames
        ]

        self.length = len(self.input_maps[0])
        if any(len(m) != self.length for m in self.input_maps):
            raise ValueError(
                f"Input and target tensors must have same length, "
                f"got input lengths {[len(m) for m in self.input_maps]}"
            )
        if any(len(m) != self.length for m in self.target_maps):
            raise ValueError(
                f"Input and target tensors must have same length, "
                f"got target lengths {[len(m) for m in self.target_maps]}"
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = [torch.from_numpy(m[index]) for m in self.input_maps]
        y = [torch.from_numpy(m[index]) for m in self.target_maps]

        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        return x, y


def convert_dict_to_npy(pt_path: str, prefix: str) -> None:
    """Convert dictionary data files to separate files

    Parameters
    ----------
    pt_path : str
        dictionary file path
    prefix : str
        new files prefix
    """
    data = torch.load(pt_path, map_location="cpu")
    for key, tensor in data.items():
        np.save(f"temp/{prefix}_{key}.npy", tensor.numpy())
