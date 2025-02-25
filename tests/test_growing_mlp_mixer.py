from unittest import TestCase, main

import torch

from gromo.containers.growing_mlp_mixer import GrowingMLPMixer
from gromo.utils.utils import global_device


class TestGrowingResidualMLP(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.input_shape = (3, 32, 32)
        self.x = torch.randn(2, *self.input_shape, device=global_device())
        self.num_classes = 10
        self.model = GrowingMLPMixer(
            input_shape=self.input_shape,
            num_features=16,
            hidden_dim_token=8,
            hidden_dim_channel=8,
            num_blocks=2,
            num_classes=self.num_classes,
        )

    def test_a_init(self):
        l1 = GrowingMLPMixer(
            input_shape=self.input_shape,
            num_features=16,
            hidden_dim_token=8,
            hidden_dim_channel=8,
            num_blocks=2,
            num_classes=self.num_classes,
        )

        self.assertIsInstance(l1, GrowingMLPMixer)
        self.assertIsInstance(l1, torch.nn.Module)

    def test_forward(self):
        with torch.no_grad():
            y = self.model(self.x)
        self.assertEqual(y.shape, (self.x.size(0), self.num_classes))


if __name__ == "__main__":
    main()
