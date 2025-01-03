"""
Module to define a two layer block similar to a BasicBlock in ResNet.
"""

import torch

from gromo.linear_growing_module import LinearAdditionGrowingModule, LinearGrowingModule

all_layer_types = {
    "linear": {"layer": LinearGrowingModule, "addition": LinearAdditionGrowingModule},
}


class GrowingResidualBlock(torch.nn.Module):
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
            in_out_features: int,
            hidden_features: int = 0,
            layer_type: str = "linear",
            activation: torch.nn.Module | None = None,
            name: str = "block",
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
        pre_activation: torch.nn.Module | None
            activation function to use before the first layer, if None use the activation function
        mid_activation: torch.nn.Module | None
            activation function to use between the two layers, if None use the activation function
        name: str
            name of the block
        kwargs_layer: dict | None
            dictionary of arguments for the layers (e.g. bias, ...)
        kwargs_first_layer: dict | None
            dictionary of arguments for the first layer, if None use kwargs_layer
        kwargs_second_layer: dict | None
            dictionary of arguments for the second layer, if None use kwargs_layer
        """
        assert layer_type in all_layer_types, f"Layer type {layer_type} not supported."
        if kwargs_layer is None:
            kwargs_layer = {}

        super(GrowingResidualBlock, self).__init__()
        self.name = name
        self.hidden_features = hidden_features

        self.activation: torch.nn.Module = activation
        self.first_layer = all_layer_types[layer_type]["layer"](
            in_out_features,
            hidden_features,
            post_layer_function=activation,
            name=f"{name}_first_layer",
            **kwargs_layer,
        )
        self.second_layer = all_layer_types[layer_type]["layer"](
            hidden_features,
            in_out_features,
            post_layer_function=torch.nn.Identity(),
            previous_module=self.first_layer,
            name=f"{name}_second_layer",
            **kwargs_layer,
        )

        self.enable_extended_forward = False
        self.eigenvalues = None
        self.parameter_update_decrease = None

        # self.activation_derivative = torch.func.grad(mid_activation)(torch.tensor(1e-5))
        # TODO: FIX this
        self.activation_derivative = 1

    @property
    def scaling_factor(self):
        return self.second_layer.scaling_factor

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
        x = self.activation(x)
        if self.hidden_features > 0:
            y, y_ext = self.first_layer.extended_forward(x)
            y, _ = self.second_layer.extended_forward(y, y_ext)
            assert (
                    _ is None
            ), f"The output of layer 2 {self.second_layer.name} should not be extended."
            del y_ext
            x = y + x
        return x

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
        x = self.activation(x)
        if self.hidden_features > 0:
            y = self.first_layer(x)
            y = self.second_layer(y)
            x = y + x
        return x

    @property
    def in_activity(self) -> torch.Tensor:
        """
        Get the input activity of the block.

        Returns
        -------
        torch.Tensor
            input activity
        """
        return self.first_layer.input

    def set_store_in_activity(self, value: bool):
        """
        Set the store_in_activity parameter of the block.
        If True, the block will store the activity after the first activation
        function.

        Parameters
        ----------
        value: bool
            value to set
        """
        self.first_layer.store_input = True

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
        self.second_layer.delete_update(include_previous=True)

    def compute_optimal_updates(
            self,
            numerical_threshold: float = 1e-15,
            statistical_threshold: float = 1e-3,
            maximum_added_neurons: int | None = None,
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
        _, _, self.parameter_update_decrease = self.second_layer.compute_optimal_delta()
        alpha, alpha_bias, _, self.eigenvalues = (
            self.second_layer.compute_optimal_added_parameters(
                numerical_threshold=numerical_threshold,
                statistical_threshold=statistical_threshold,
                maximum_added_neurons=maximum_added_neurons,
                update_previous=True,
            )
        )

    def apply_change(self) -> None:
        """
        Apply the optimal delta and extend the layer with current
        optimal delta and layer extension with the current scaling factor.
        """
        self.second_layer.apply_change(apply_previous=True)
        self.hidden_features += self.eigenvalues.shape[0]

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
        self.second_layer.sub_select_optimal_added_parameters(keep_neurons, sub_select_previous=True)

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """
        Get the first order improvement of the block.

        Returns
        -------
        torch.Tensor
            first order improvement
        """
        return (
                self.parameter_update_decrease
                + self.activation_derivative * (self.eigenvalues ** 2).sum()
        )


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)

    reduction = "mean"


    def train(model, dataset, optimizer):
        total_loss = 0
        for x, y in dataset:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y, reduction=reduction)
            total_loss += loss
            loss.backward()
            optimizer.step()
        return total_loss / len(dataset)


    def evaluate(model, dataset):
        total_loss = 0
        with torch.no_grad():
            for x, y in dataset:
                y_hat = model(x)
                loss = F.mse_loss(y_hat, y, reduction=reduction)
                total_loss += loss
        return total_loss / len(dataset)


    def compute_statistics(model, dataset):
        model.init_computation()
        total_loss = 0
        for x, y in dataset:
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y, reduction="sum")
            total_loss += loss
            loss.backward()
            model.update_computation()
        return total_loss / len(dataset)


    def evaluate_with_extension(model, dataset, scaling_factor):
        model.second_layer.scaling_factor = scaling_factor
        total_loss = 0
        with torch.no_grad():
            for x, y in dataset:
                y_hat = model.extended_forward(x)
                loss = F.mse_loss(y_hat, y, reduction=reduction)
                total_loss += loss
        return total_loss / len(dataset)


    # Define the dataset
    batch_size = 100
    num_batch = 10
    in_features = 10
    dataset = [(torch.randn(batch_size, in_features), torch.randn(batch_size, in_features)) for _ in range(num_batch)]

    # Define the block
    hidden_features = 2
    block = GrowingResidualBlock(in_features, hidden_features, activation=nn.ReLU(), name="block")
    print(block)

    # Define the optimizer
    optimizer = torch.optim.SGD(block.parameters(), lr=0.01 / num_batch, momentum=0.9, weight_decay=1e-3)

    # Regular training
    for epoch in range(42):
        training_loss = train(block, dataset, optimizer)
        print(f'Epoch {epoch}, Training Loss: {training_loss}')

    # Training loss after training
    training_loss_after = evaluate(block, dataset)
    print(f'Training Loss after training: {training_loss_after}')

    # Gathering growing statistics
    statistics_loss = compute_statistics(block, dataset)
    print(f'Training Loss after gathering statistics: {statistics_loss}')

    # Compute the optimal update
    keep_neurons = 1
    block.compute_optimal_updates(maximum_added_neurons=keep_neurons)

    print(block.eigenvalues)

    # Training loss with the change
    scaling_factor = 0.5
    loss_with_extension = evaluate_with_extension(block, dataset, scaling_factor)
    print(f'Training Loss with the change: {loss_with_extension}')
    print(f'First order improvement: {block.first_order_improvement / (batch_size * num_batch)}')
    print(f'Zero-th order improvement: {training_loss_after - loss_with_extension}')

    # Apply the change
    print('Apply the change')
    block.apply_change()

    # Delete the update
    print('Delete the update')
    block.delete_update()
    block.reset_computation()

    # Training loss after the change
    training_loss_after_change = evaluate(block, dataset)
    print(f'Training Loss after the change: {training_loss_after_change}')
    print(f'Zero-th order improvement: {training_loss_after - training_loss_after_change}')

    print(block)

    # assert loss_with_extension == training_loss_after_change

    # Tolerance for floating-point comparisons
    tolerance = 1e-6  # Adjust this value based on the required precision

    # Assert the two values are "close enough" within the tolerance
    assert torch.isclose(
        loss_with_extension,
        training_loss_after_change,
        atol=tolerance
    ), (
        f"Loss with extension ({loss_with_extension}) "
        f"and training loss after change ({training_loss_after_change}) "
        f"are not close enough. (Absolute difference: {torch.abs(loss_with_extension - training_loss_after_change)})"
    )
