<p align=center>
  <img alt="banner" src="docs/source/images/logo_bg_black.png" height=500px>
</p>

[![codecov](https://codecov.io/github/growingnet/gromo/graph/badge.svg?token=87HWKJ6H6D)](https://codecov.io/github/growingnet/gromo)
[![Tests](https://github.com/growingnet/gromo/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/growingnet/gromo/actions/workflows/tests.yml)

# GroMo

Gromo is a collaborative effort for designing efficient and growable networks
for machine learning. It is a Python package that provides a set of tools for
pytorch users to train neural networks, that grow in size and complexity as
they learn. The package is designed to be modular and flexible, allowing users
to easily design and train their own networks.

The package is built on top of `torch.nn.Module`,
examples shows how to use the package to train a simple neural network, and how
to grow the network as it learns. The theoretical and algorithmic details of
Gromo are described in the paper indicated below.

The package is still in development, if you would like to contribute, please
contact us!

```
Verbockhaven, M., Rudkiewicz, T., Chevallier, S., and Charpiat, G. (2024). Growing tiny networks: Spotting expressivity bottlenecks and fixing them optimally. Transactions on Machine Learning Research.
```
