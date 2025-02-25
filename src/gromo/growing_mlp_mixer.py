import torch.nn
from torch import nn

from gromo.linear_growing_module import LinearGrowingModule
from gromo.utils.utils import global_device


class GrowingMLPBlock(nn.Module):
    """
    Represents a block of a growing network.

    Sequence of layers:
    - Activation pre
    - Layer first
    - Activation mid
    - Layer second
    """

    def __init__(
        self,
        num_features: int,
        hidden_features: int = 0,
        dropout: float = 0.0,
        name: str | None = None,
        kwargs_layer: dict | None = None,
    ) -> None:
        """
        Initialise the block.

        Parameters
        ----------
        in_out_features: int
            number of input and output features, in cas of convolutional layer, the number of channels
        hidden_features: int
            number of hidden features, if zero the block is the zero function
        layer_type: str
            type of layer to use either "linear" or "conv"
        activation: torch.nn.Module | None
            activation function to use, if None use the identity function
        name: str
            name of the block
        kwargs_layer: dict | None
            dictionary of arguments for the layers (e.g. bias, ...)
        """
        if kwargs_layer is None:
            kwargs_layer = {}

        super(GrowingMLPBlock, self).__init__()
        self.name = name

        self.first_layer = LinearGrowingModule(
            num_features,
            hidden_features,
            post_layer_function=nn.GELU(),
            **kwargs_layer,
        )
        self.dropout = nn.Dropout(dropout)
        self.second_layer = LinearGrowingModule(
            hidden_features,
            num_features,
            post_layer_function=nn.Identity(),
            previous_module=self.first_layer,
            **kwargs_layer,
        )
        self.enable_extended_forward = False

        # self.activation_derivative = torch.func.grad(self.first_layer.post_layer_function)(torch.tensor(1e-5))
        # TODO: FIX this
        # self.activation_derivative = 1

    def __setattr__(self, key, value):
        if key in [
            "scaling_factor",
            "eigenvalues_extension",
            "parameter_update_decrease",
            "first_order_improvement",
        ]:
            self.second_layer.__setattr__(key, value)
        else:
            nn.Module.__setattr__(self, key, value)

    @property
    def hidden_features(self):
        return self.second_layer.in_features

    @property
    def scaling_factor(self):
        return self.second_layer.scaling_factor

    @property
    def eigenvalues_extension(self):
        return self.second_layer.eigenvalues_extension

    @property
    def parameter_update_decrease(self):
        return self.second_layer.parameter_update_decrease

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """
        Get the first order improvement of the block.

        Returns
        -------
        torch.Tensor
            first order improvement
        """
        return self.second_layer.first_order_improvement

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block with the current modifications.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        y, y_ext = self.first_layer.extended_forward(x)
        y = self.dropout(y)
        if y_ext is not None:
            y_ext = self.dropout(y_ext)
        y, _ = self.second_layer.extended_forward(y, y_ext)

        assert (
            _ is None
        ), f"The output of layer 2 {self.second_layer.name} should not be extended."
        del y_ext

        y = self.dropout(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        y = self.first_layer(x)
        y = self.dropout(y)
        y = self.second_layer(y)
        y = self.dropout(y)
        return y

    def init_computation(self):
        """
        Initialise the computation of the block.
        """
        self.first_layer.init_computation()
        self.second_layer.init_computation()

    def update_computation(self, desired_activation: torch.Tensor | None = None):
        """
        Update the computation of the block.

        Parameters
        ----------
        desired_activation: torch.Tensor
            desired direction of the output variation of the block
        """
        self.first_layer.update_computation()
        self.second_layer.update_computation()

    def reset_computation(self):
        """
        Reset the computation of the block.
        """
        self.first_layer.reset_computation()
        self.second_layer.reset_computation()

    def delete_update(self):
        """
        Delete the update of the block.
        """
        self.first_layer.delete_update()
        self.second_layer.delete_update(include_previous=True)

    def compute_optimal_update(
        self,
        part: str = "all",
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Compute the optimal update for second layer and additional neurons.

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root of the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        """
        assert part in [
            "all",
            "parameter",
            "neuron",
        ], f"{part=} should be in ['all', 'parameter', 'neuron']"

        if part == "parameter":
            _, _, _ = self.second_layer.compute_optimal_delta(dtype=dtype)
        elif part == "neuron":
            _, _ = self.second_layer.compute_optimal_updates(
                zero_delta=True,
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
                update_previous=True,
                dtype=dtype,
            )
            self.second_layer.optimal_delta_layer = None
            self.second_layer.parameter_update_decrease = 0
        elif part == "all":
            _, _ = self.second_layer.compute_optimal_updates(
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
                update_previous=True,
                dtype=dtype,
            )

    def apply_change(self) -> None:
        """
        Apply the optimal delta and extend the layer with current
        optimal delta and layer extension with the current scaling factor.
        """
        self.second_layer.apply_change(apply_previous=True)

    def sub_select_optimal_added_parameters(
        self,
        keep_neurons: int,
    ) -> None:
        """
        Select the first keep_neurons neurons of the optimal added parameters.

        Parameters
        ----------
        keep_neurons: int
            number of neurons to keep
        """
        self.eigenvalues = self.eigenvalues[:keep_neurons]
        self.second_layer.sub_select_optimal_added_parameters(
            keep_neurons, sub_select_previous=True
        )

    def number_of_parameters(self):
        num_param = self.first_layer.number_of_parameters()
        num_param += self.second_layer.number_of_parameters()
        return num_param

    @staticmethod
    def tensor_statistics(tensor) -> dict[str, float]:
        min_value = tensor.min().item()
        max_value = tensor.max().item()
        mean_value = tensor.mean().item()
        if tensor.numel() > 1:
            std_value = tensor.std().item()
        else:
            std_value = -1
        return {
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "std": std_value,
        }

    def weights_statistics(self) -> dict[int, dict[str, dict[str, float] | int]]:
        statistics = {}
        for i, layer in enumerate([self.first_layer, self.second_layer]):
            if layer.weight.numel() == 0:
                continue
            statistics[i] = {
                "weight": self.tensor_statistics(layer.weight),
            }
            if layer.bias is not None:
                statistics[i]["bias"] = self.tensor_statistics(layer.bias)
            statistics[i]["input_shape"] = layer.in_features
            statistics[i]["output_shape"] = layer.out_features
        return statistics

    def update_information(self):
        layer_information = dict()
        layer_information["update_value"] = self.first_order_improvement
        layer_information["parameter_improvement"] = self.parameter_update_decrease
        layer_information["eigenvalues_extension"] = self.eigenvalues_extension
        layer_information["scaling_factor"] = self.scaling_factor
        layer_information["added_neurons"] = (
            0
            if self.eigenvalues_extension is None
            else self.eigenvalues_extension.size(0)
        )
        return layer_information


__growing_methods__ = [
    "init_computation",
    "reset_computation",
    "update_computation",
    "delete_update",
    "compute_optimal_update",
    "apply_change",
    "number_of_parameters",
    "weights_statistics",
    "update_information",
]

__growing_attributes__ = [
    "scaling_factor",
    "eigenvalues_extension",
    "parameter_update_decrease",
    "first_order_improvement",
]


class GrowingTokenMixer(nn.Module):
    def __init__(self, num_patches, num_features, hidden_features, dropout, name=None):
        super(GrowingTokenMixer, self).__init__()
        self.norm = nn.LayerNorm(num_features, device=global_device())
        self.mlp = GrowingMLPBlock(num_patches, hidden_features, dropout)

    def __getattr__(self, item):
        if item in __growing_attributes__:
            return getattr(self.mlp, item)
        elif item != "number_of_parameters" and item in __growing_methods__:
            return getattr(self.mlp, item)
        else:
            return super(GrowingTokenMixer, self).__getattr__(item)

    def __setattr__(self, key, value):
        if key in __growing_attributes__:
            self.mlp.__setattr__(key, value)
        else:
            nn.Module.__setattr__(self, key, value)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_features, num_patches)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block with the current modifications.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        residual = x
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.mlp.extended_forward(y)
        y = y.transpose(1, 2)
        out = y + residual
        return out

    def number_of_parameters(self):
        return self.mlp.number_of_parameters() + sum(
            p.numel() for p in self.norm.parameters()
        )


class GrowingChannelMixer(nn.Module):
    def __init__(self, num_features, hidden_features, dropout, name=None):
        super(GrowingChannelMixer, self).__init__()
        self.norm = nn.LayerNorm(num_features, device=global_device())
        self.mlp = GrowingMLPBlock(num_features, hidden_features, dropout)

    def __getattr__(self, item):
        if item in __growing_attributes__:
            return getattr(self.mlp, item)
        elif item != "number_of_parameters" and item in __growing_methods__:
            return getattr(self.mlp, item)
        else:
            return super(GrowingChannelMixer, self).__getattr__(item)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block with the current modifications.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        residual = x
        x = self.norm(x)
        x = self.mlp.extended_forward(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out

    def number_of_parameters(self):
        return self.mlp.number_of_parameters() + sum(
            p.numel() for p in self.norm.parameters()
        )


class GrowingMixerLayer(nn.Module):
    def __init__(
        self,
        num_patches,
        num_features,
        hidden_dim_token,
        hidden_dim_channel,
        dropout,
        name="Mixer Layer",
    ):
        super(GrowingMixerLayer, self).__init__()
        self.token_mixer = GrowingTokenMixer(
            num_patches, num_features, hidden_dim_token, dropout
        )
        self.channel_mixer = GrowingChannelMixer(
            num_features, hidden_dim_channel, dropout
        )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, num_patches, num_features)
        return x

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block with the current modifications.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        x = self.token_mixer.extended_forward(x)
        x = self.channel_mixer.extended_forward(x)
        return x

    def init_computation(self):
        """
        Initialise the computation of the block.
        """
        self.token_mixer.init_computation()
        self.channel_mixer.init_computation()

    def reset_computation(self):
        """
        Reset the computation of the block.
        """
        self.token_mixer.reset_computation()
        self.channel_mixer.reset_computation()

    def update_computation(self):
        """
        Update the computation of the block.
        """
        self.token_mixer.update_computation()
        self.channel_mixer.update_computation()

    def delete_update(self):
        """
        Delete the update of the block.
        """
        self.token_mixer.delete_update()
        self.channel_mixer.delete_update()

    def compute_optimal_update(
        self,
        part: str = "all",
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Compute the optimal update for second layer and additional neurons.

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root of the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        """
        self.token_mixer.compute_optimal_update(
            part=part,
            numerical_threshold=numerical_threshold,
            statistical_threshold=statistical_threshold,
            maximum_added_neurons=maximum_added_neurons,
            dtype=dtype,
        )
        self.channel_mixer.compute_optimal_update(
            part=part,
            numerical_threshold=numerical_threshold,
            statistical_threshold=statistical_threshold,
            maximum_added_neurons=maximum_added_neurons,
            dtype=dtype,
        )

    def apply_change(self) -> None:
        """
        Apply the optimal delta and extend the layer with current
        optimal delta and layer extension with the current scaling factor.
        """
        self.token_mixer.apply_change()
        self.channel_mixer.apply_change()

    def number_of_parameters(self):
        num_param = self.token_mixer.number_of_parameters()
        num_param += self.channel_mixer.number_of_parameters()
        return num_param

    def weights_statistics(self) -> dict[int, dict[str, dict[str, float] | int]]:
        statistics = {}
        statistics[0] = self.token_mixer.weights_statistics()
        statistics[1] = self.channel_mixer.weights_statistics()
        return statistics

    def update_information(self):
        layer_information = dict()
        layer_information["token_mixer"] = self.token_mixer.update_information()
        layer_information["channel_mixer"] = self.channel_mixer.update_information()
        return layer_information


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches**2
    return num_patches


class GrowingMLPMixer(nn.Module):
    def __init__(
        self,
        input_shape=(3, 32, 32),
        patch_size=4,
        num_features=128,
        hidden_dim_token=64,
        hidden_dim_channel=512,
        num_blocks=8,
        num_classes=10,
        dropout=0.0,
    ):
        in_channels, image_size, _ = input_shape
        num_patches = check_sizes(image_size, patch_size)
        super(GrowingMLPMixer, self).__init__()
        # per-patch fully-connected is equivalent to strided conv2d
        self.patcher = nn.Conv2d(
            in_channels,
            num_features,
            kernel_size=patch_size,
            stride=patch_size,
            device=global_device(),
        )
        self.mixers: nn.ModuleList[GrowingMixerLayer] = nn.ModuleList(
            [
                GrowingMixerLayer(
                    num_patches,
                    num_features,
                    hidden_dim_token,
                    hidden_dim_channel,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.classifier = nn.Linear(num_features, num_classes, device=global_device())
        self.currently_updated_block = None

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = patches
        for mixer in self.mixers:
            embedding = mixer(embedding)
        # embedding.shape == (batch_size, num_patches, num_features)
        embedding = embedding.mean(dim=1)
        logits = self.classifier(embedding)
        return logits

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block with the current modifications.

        Parameters
        ----------
        x: torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output tensor
        """
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = patches
        for mixer in self.mixers:
            embedding = mixer.extended_forward(embedding)
        # embedding.shape == (batch_size, num_patches, num_features)
        embedding = embedding.mean(dim=1)
        logits = self.classifier(embedding)
        return logits

    def init_computation(self):
        """
        Initialise the computation of the block.
        """
        for mixer in self.mixers:
            mixer.init_computation()

    def reset_computation(self):
        """
        Reset the computation of the block.
        """
        for mixer in self.mixers:
            mixer.reset_computation()

    def update_computation(self):
        """
        Update the computation of the block.
        """
        for mixer in self.mixers:
            mixer.update_computation()

    def delete_update(self):
        """
        Delete the update of the block.
        """
        for mixer in self.mixers:
            mixer.delete_update()
        self.currently_updated_block = None

    def compute_optimal_update(
        self,
        part: str = "all",
        numerical_threshold: float = 1e-15,
        statistical_threshold: float = 1e-3,
        maximum_added_neurons: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Compute the optimal update for second layer and additional neurons.

        Parameters
        ----------
        numerical_threshold: float
            threshold to consider an eigenvalue as zero in the square root of the inverse of S
        statistical_threshold: float
            threshold to consider an eigenvalue as zero in the SVD of S{-1/2} N
        maximum_added_neurons: int | None
            maximum number of added neurons, if None all significant neurons are kept
        """
        for mixer in self.mixers:
            mixer.compute_optimal_update(
                part=part,
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
                dtype=dtype,
            )

    def apply_change(self) -> None:
        """
        Apply the optimal delta and extend the layer with current
        optimal delta and layer extension with the current scaling factor.
        """
        self.currently_updated_block.apply_change()
        self.currently_updated_block = None

    def number_of_parameters(self):
        num_param = sum(p.numel() for p in self.patcher.parameters())
        for mixer in self.mixers:
            num_param += mixer.number_of_parameters()
        num_param += sum(p.numel() for p in self.classifier.parameters())
        return num_param

    def weights_statistics(self) -> dict[int, dict[str, dict[str, float] | int]]:
        statistics = {}
        for i, mixer in enumerate(self.mixers):
            statistics[i] = mixer.weights_statistics()
        return statistics

    def update_information(self):
        assert self.currently_updated_block is not None, "No block is currently updated."
        return self.currently_updated_block.update_information()

    def select_best_update(self):
        token_mixers_first_order_improvement = torch.tensor(
            [mixer.token_mixer.first_order_improvement for mixer in self.mixers]
        )
        channel_mixers_first_order_improvement = torch.tensor(
            [mixer.channel_mixer.first_order_improvement for mixer in self.mixers]
        )
        best_token_mixer_index = torch.argmax(token_mixers_first_order_improvement)
        best_channel_mixer_index = torch.argmax(channel_mixers_first_order_improvement)

        best_token_mixer_improvement = token_mixers_first_order_improvement[
            best_token_mixer_index
        ]
        best_channel_mixer_improvement = channel_mixers_first_order_improvement[
            best_channel_mixer_index
        ]
        token_or_channels = (
            "token"
            if best_token_mixer_improvement > best_channel_mixer_improvement
            else "channel"
        )

        for i, mixer in enumerate(self.mixers):
            if token_or_channels == "token":
                if i != best_token_mixer_index:
                    mixer.delete_update()
                else:
                    mixer.channel_mixer.delete_update()
                    self.currently_updated_block = mixer.token_mixer.mlp.second_layer
                    print(f"Selected token mixer {i}")
            elif token_or_channels == "channel":
                if i != best_channel_mixer_index:
                    mixer.delete_update()
                else:
                    mixer.token_mixer.delete_update()
                    self.currently_updated_block = mixer.channel_mixer.mlp.second_layer
                    print(f"Selected channel mixer {i}")

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """
        Get the first order improvement of the block.

        Returns
        -------
        torch.Tensor
            first order improvement
        """
        return self.currently_updated_block.first_order_improvement


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # define the input shape
    img_size = 32
    input_shape = (3, img_size, img_size)
    # define the number of features
    num_features = 32
    # define the number of hidden token features
    hidden_token_features = 8
    # define the number of hidden channel features
    hidden_channel_features = 16
    # define the number of output features
    num_classes = 10
    # define the number of blocks
    num_blocks = 1

    # create the growing residual MLP
    model = GrowingMLPMixer(
        input_shape=input_shape,
        num_features=num_features,
        hidden_dim_token=hidden_token_features,
        hidden_dim_channel=hidden_channel_features,
        num_blocks=num_blocks,
        num_classes=num_classes,
    )

    # print the model
    print(model)

    # define the input tensor
    x = torch.rand((2, *input_shape))
    print(f"Input shape: {x.shape}")
    # forward pass
    y = model(x)
    print(f"Output shape: {y.shape}")
