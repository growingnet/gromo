"""
=============================
Minimal Linear Growing Layers
=============================

This example shows minimal linear growing layers.
"""

# Authors: Theo Rudkiewicz <theo.rudkiewicz@inria.fr>
#          Sylvain Chevallier <sylvain.chevallier@universite-paris-saclay.fr>

###############################################################################
# Setup
# -----
# Importing the modules

import torch

from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.utils import global_device


###############################################################################
# Define three linear growing layers of size 1, 1, 1 with ReLU activation

l1 = LinearGrowingModule(
    1, 1, use_bias=True, post_layer_function=torch.nn.ReLU(), name="l1"
)
l2 = LinearGrowingModule(
    1,
    1,
    use_bias=True,
    previous_module=l1,
    post_layer_function=torch.nn.ReLU(),
    name="l2",
)
l3 = LinearGrowingModule(1, 1, use_bias=True, previous_module=l2, name="l3")

###############################################################################
# Generate random data, initialize the computation and compute optimal updates

x = torch.randn(200, 1, device=global_device())
net = torch.nn.Sequential(l1, l2, l3)

print(net)

for layer in net:
    layer.init_computation()

for layer in net:
    print(layer.__str__(verbose=1))

y = net(x)
loss = torch.norm(y)
print(f"loss: {loss}")
loss.backward()

for layer in net:
    layer.update_computation()

    layer.compute_optimal_updates()

for layer in net:
    layer.reset_computation()

l1.delete_update()
l3.delete_update()

l2.scaling_factor = 1

###############################################################################
# Print parameters before and after applying the optimal update

print(f"{l2.first_order_improvement=}")
print(f"{l2.weight=}")
print(f"{l2.bias=}")
print(f"{l2.optimal_delta_layer=}")
print(f"{l2.parameter_update_decrease=}")
print(f"{l2.extended_input_layer=}")
print(f"{l2.extended_input_layer.weight=}")
print(f"{l2.extended_input_layer.bias=}")
print(f"{l1.extended_output_layer=}")
print(f"{l2.eigenvalues_extension=}")

x_ext = None
for layer in net:
    x, x_ext = layer.extended_forward(x, x_ext)

new_loss = torch.norm(x)
print(f"loss: {new_loss}, {loss - new_loss} improvement")
l2.apply_change()

print("------- New weights -------")
print(f"{l1.weight=}")
print(f"{l2.weight=}")
print(f"{l3.weight=}")
print("------- New biases -------")
print(f"{l1.bias=}")
print(f"{l2.bias=}")
print(f"{l3.bias=}")

for layer in net:
    layer.init_computation()

for layer in net:
    print(layer.__str__(verbose=2))

y = net(x)
loss = torch.norm(y)
print(f"loss: {loss}")
loss.backward()

for layer in net:
    layer.update_computation()

    layer.compute_optimal_updates()
