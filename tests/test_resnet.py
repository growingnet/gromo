import torch

from gromo.containers.resnet import ResNetBasicBlock, init_full_resnet_structure
from gromo.utils.utils import global_device


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

    def test_forward_backward(self):
        """Test forward and backward pass of the ResNet model."""
        model = init_full_resnet_structure(
            input_shape=(3, 32, 32),
            out_features=10,
            number_of_blocks_per_stage=2,
            reduction_factor=0.5,
        )
        x = torch.randn(4, 3, 32, 32, device=global_device())
        output = model(x)
        self.assertShapeEqual(output, (4, 10), "Output shape should be (4, 10)")
        loss = output.sum()
        loss.backward()  # Check that backward pass works without error

    def test_append_block(self):
        """
        Test appending blocks with 0 and non-zero features and verify forward
        behavior.
        """
        # Use CPU device for testing
        device = torch.device("cpu")

        # Create a simple network with small input
        model = init_full_resnet_structure(
            input_shape=(3, 32, 32),
            out_features=10,
            number_of_blocks_per_stage=1,
            reduction_factor=0.5,
            device=device,
        )

        # Create a random input on the same device
        x = torch.randn(2, 3, 32, 32, device=device)

        # Initial forward pass
        output1 = model(x)
        self.assertShapeEqual(output1, (2, 10))

        # Verify initial number of blocks in each stage
        for stage_idx, stage in enumerate(model.stages):  # type: ignore
            self.assertEqual(
                len(stage), 1, f"Stage {stage_idx} should initially have 1 block"
            )

        # Add a block with 0 hidden features to stage 0
        model.append_block(stage_index=0, hidden_channels=0)

        # Verify number of blocks increased
        self.assertEqual(len(model.stages[0]), 2, "Stage 0 should now have 2 blocks")  # type: ignore
        for stage_idx in range(1, 4):
            self.assertEqual(
                len(model.stages[stage_idx]),  # type: ignore
                1,
                f"Stage {stage_idx} should still have 1 block",
            )

        # Forward pass after adding block with 0 features
        # The output should be the same since the block has 0 hidden features
        output2 = model(x)
        self.assertShapeEqual(output2, (2, 10))
        self.assertAllClose(
            output1,
            output2,
            message="Output should not change when adding block with 0 features",
        )

        # Add a block with non-zero hidden features to stage 1
        model.append_block(stage_index=1, hidden_channels=32)

        # Verify number of blocks increased
        self.assertEqual(len(model.stages[0]), 2, "Stage 0 should still have 2 blocks")  # type: ignore
        self.assertEqual(len(model.stages[1]), 2, "Stage 1 should now have 2 blocks")  # type: ignore
        for stage_idx in range(2, 4):
            self.assertEqual(
                len(model.stages[stage_idx]),  # type: ignore
                1,
                f"Stage {stage_idx} should still have 1 block",
            )

        # Forward pass after adding block with features
        # The output should change since the block has non-zero hidden features
        output3 = model(x)
        self.assertShapeEqual(output3, (2, 10))

        # Check that outputs are different (with high probability due to random init)
        # We use a small tolerance to allow for numerical errors but expect difference
        self.assertFalse(
            torch.allclose(output2, output3, atol=1e-6, rtol=1e-5),
            "Output should change when adding block with non-zero features",
        )

        # Test invalid stage index
        with self.assertRaises(IndexError):
            model.append_block(stage_index=10, hidden_channels=64)

        with self.assertRaises(IndexError):
            model.append_block(stage_index=-1, hidden_channels=64)
