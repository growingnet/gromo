import torch

from gromo.containers.resnet import ResNetBasicBlock, init_full_resnet_structure


try:
    from tests.torch_unittest import TorchTestCase
except ImportError:
    from torch_unittest import TorchTestCase


class TestResNet(TorchTestCase):
    """Test ResNet implementation."""

    def test_init_full_resnet_structure_variations(self):
        """
        Test init_full_resnet_structure with different settings to cover all
        branches.
        """
        # Test 1: Basic instantiation with default parameters
        model = init_full_resnet_structure()
        self.assertIsInstance(model, ResNetBasicBlock)
        self.assertEqual(len(model.stages), 4)
        # Default number_of_blocks_per_stage is 2, so each stage should have 2 blocks
        for stage_idx, stage in enumerate(model.stages):  # type: ignore
            self.assertEqual(len(stage), 2, f"Stage {stage_idx} should have 2 blocks")

        # Test 2: Small inputs (should be auto-detected from input_shape)
        model_small = init_full_resnet_structure(
            input_shape=(3, 32, 32),
            out_features=10,
        )
        self.assertIsInstance(model_small, ResNetBasicBlock)
        self.assertTrue(model_small.small_inputs)

        # Test 3: Large inputs (should be auto-detected)
        model_large = init_full_resnet_structure(
            input_shape=(3, 224, 224),
            out_features=1000,
        )
        self.assertIsInstance(model_large, ResNetBasicBlock)
        self.assertFalse(model_large.small_inputs)

        # Test 4: Explicit small_inputs=True
        model_explicit_small = init_full_resnet_structure(
            input_shape=(3, 224, 224),
            small_inputs=True,
            out_features=100,
        )
        self.assertIsInstance(model_explicit_small, ResNetBasicBlock)
        self.assertTrue(model_explicit_small.small_inputs)

        # Test 5: Explicit small_inputs=False
        model_explicit_large = init_full_resnet_structure(
            input_shape=(3, 32, 32),
            small_inputs=False,
            out_features=10,
        )
        self.assertIsInstance(model_explicit_large, ResNetBasicBlock)
        self.assertFalse(model_explicit_large.small_inputs)

        # Test 6: Custom number_of_blocks_per_stage as integer
        model_custom_blocks_int = init_full_resnet_structure(
            input_shape=(3, 32, 32),
            number_of_blocks_per_stage=3,
        )
        self.assertIsInstance(model_custom_blocks_int, ResNetBasicBlock)
        for stage_idx, stage in enumerate(model_custom_blocks_int.stages):  # type: ignore
            stage: torch.nn.Sequential
            self.assertEqual(len(stage), 3, f"Stage {stage_idx} should have 3 blocks")

        # Test 7: Different blocks per stage
        model_varied_blocks = init_full_resnet_structure(
            input_shape=(3, 32, 32),
            number_of_blocks_per_stage=(1, 2, 3, 4),
        )
        self.assertIsInstance(model_varied_blocks, ResNetBasicBlock)
        expected_blocks = [1, 2, 3, 4]
        for stage_idx, stage in enumerate(model_varied_blocks.stages):  # type: ignore
            self.assertEqual(
                len(stage),
                expected_blocks[stage_idx],
                f"Stage {stage_idx} should have {expected_blocks[stage_idx]} blocks",
            )

        # Test 8: Explicit in_features (overrides input_shape channels)
        model_explicit_in = init_full_resnet_structure(
            input_shape=(3, 32, 32),
            in_features=1,
        )
        self.assertIsInstance(model_explicit_in, ResNetBasicBlock)

        # Test 9: input_shape as torch.Size
        model_torch_size = init_full_resnet_structure(
            input_shape=torch.Size((3, 32, 32)),  # type: ignore
        )
        self.assertIsInstance(model_torch_size, ResNetBasicBlock)

        # Test 10: Wrong length tuple should raise error
        with self.assertRaises(TypeError):
            init_full_resnet_structure(
                input_shape=(3, 32, 32),
                number_of_blocks_per_stage=(2, 2, 2),  # type: ignore
            )
