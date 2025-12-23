from typing import Any

import torch
import torch.utils.data


class SyntheticDataloader(torch.utils.data.DataLoader):
    r"""
    An abstract dataloader class for generating synthetic data.
    """

    def __init__(
        self,
        nb_sample: int = 1,
        batch_size: int = 100,
        seed: int = 0,
        device: torch.device | None = None,
        in_features: int = 1,
        out_features: int = 1,
    ):
        self.nb_sample = nb_sample
        self.batch_size: int = batch_size  # type: ignore
        self.seed = seed
        self.sample_index = 0
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.in_features = in_features
        self.out_features = out_features

    def __iter__(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        torch.manual_seed(self.seed)
        self.sample_index = 0
        return self

    def __next__(self) -> Any:
        if self.sample_index >= self.nb_sample:
            raise StopIteration
        self.sample_index += 1

    def __len__(self):
        return self.nb_sample


class SinDataloader(SyntheticDataloader):
    r"""
    A simple dataloader that generates batches of synthetic data for the function
    y = sin(x), where x is uniformly sampled from [0, 2\pi].
    """

    def __next__(self):
        super().__next__()
        x = torch.rand(self.batch_size, 1, device=self.device) * 2 * torch.pi
        y = torch.sin(x)
        return x, y


class MultiSinDataloader(SyntheticDataloader):
    r"""
    A simple dataloader that generates batches of synthetic data for the
    function y[d] = sum_{i=1}^{k} sin(i * x[i] + d), where x is sampled
    from a gaussian distribution.
    """

    def __init__(
        self,
        nb_sample: int = 1,
        batch_size: int = 100,
        seed: int = 0,
        in_features: int = 3,
        out_features: int = 1,
        device: torch.device | None = None,
    ):
        super().__init__(
            nb_sample=nb_sample,
            batch_size=batch_size,
            seed=seed,
            device=device,
            in_features=in_features,
            out_features=out_features,
        )

    def __next__(self):
        super().__next__()
        x = torch.randn(self.batch_size, self.in_features, device=self.device)
        y = torch.empty(self.batch_size, self.out_features, device=self.device)
        for d in range(self.out_features):
            y[:, d] = sum(
                torch.sin((i + 1) * x[:, i] + d) for i in range(self.in_features)
            )
        return x, y
