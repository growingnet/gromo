from gromo.containers.resnet import (
    init_full_resnet_structure,
)


try:
    from tests.torch_unittest import TorchTestCase
except ImportError:
    from torch_unittest import TorchTestCase


class TestWideResNet(TorchTestCase):
    def test_wrn28_1(self):
        """
        Compare the number of parameters in our implementation of WRN-28-1-dropout
        with a reference implementation from
        https://github.com/bmsookim/wide-resnet.pytorch.git
        on CIFAR-10 (3 input channels, 10 output classes).
        """
        model = init_full_resnet_structure(
            number_of_blocks_per_stage=4,
            nb_stages=3,
            dropout_rate=0.3,
            small_inputs=True,
            reduction_factor=1,
            inplanes=16,
            out_features=10,
        )

        n_params = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(n_params, 370506, f"Expected 370506 parameters, got {n_params}")
