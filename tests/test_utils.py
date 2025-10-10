import random
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from gromo.utils.utils import (
    activation_fn,
    batch_gradient_descent,
    calculate_true_positives,
    compute_tensor_stats,
    evaluate_dataset,
    f1,
    f1_macro,
    f1_micro,
    get_correct_device,
    global_device,
    line_search,
    mini_batch_gradient_descent,
    reset_device,
    safe_forward,
    set_device,
    set_from_conf,
    torch_ones,
    torch_zeros,
)
from tests.torch_unittest import TorchTestCase


class TestUtils(TorchTestCase):
    @classmethod
    def setUpClass(cls):
        """Set up available devices for testing"""
        cls.available_devices = ["cpu"]
        if torch.cuda.is_available():
            cls.available_devices.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cls.available_devices.append("mps")

    def test_set_device(self) -> None:
        # Test setting each available device
        for device_name in self.available_devices:
            with self.subTest(device=device_name):
                set_device(device_name)
                self.assertEqual(global_device(), torch.device(device_name))

                # Test with torch.device object
                set_device(torch.device(device_name))
                self.assertEqual(global_device(), torch.device(device_name))

    def test_reset_device(self) -> None:
        """Test device reset functionality"""
        # First set a specific device (use first available)
        set_device(self.available_devices[0])
        self.assertEqual(global_device(), torch.device(self.available_devices[0]))

        # Test reset with CUDA available
        with patch("torch.cuda.is_available", return_value=True):
            reset_device()
            self.assertEqual(global_device(), torch.device("cuda"))

        # Test reset with CUDA not available
        with patch("torch.cuda.is_available", return_value=False):
            reset_device()
            self.assertEqual(global_device(), torch.device("cpu"))

    def test_get_correct_device(self) -> None:
        """Test device precedence logic"""
        # Create a mock object with config data
        mock_obj = MagicMock()

        # Test with each available device
        for device_name in self.available_devices:
            with self.subTest(device=device_name):
                # Test with explicit device argument (highest precedence)
                mock_obj._config_data = {}
                device = get_correct_device(mock_obj, device=device_name)
                self.assertEqual(device, torch.device(device_name))

                # Test with None device - should use config
                mock_obj._config_data = {"device": device_name}
                device = get_correct_device(mock_obj, None)
                self.assertEqual(device, torch.device(device_name))

                # Test with None device and empty config - should use global device
                original_device = global_device()
                set_device(device_name)  # Set to current device
                mock_obj._config_data = {}
                device = get_correct_device(mock_obj, None)
                self.assertEqual(device, global_device())
                set_device(original_device)  # Reset to original device

    def test_torch_zeros(self) -> None:
        # Test on each available device
        for device_name in self.available_devices:
            with self.subTest(device=device_name):
                set_device(device_name)
                size = (random.randint(1, 10), random.randint(1, 10))
                tensor = torch_zeros(size)
                tensor_device = torch.device(
                    "cuda" if tensor.is_cuda else "mps" if tensor.is_mps else "cpu"
                )
                self.assertEqual(tensor_device, global_device())
                self.assertEqual(tensor.shape, size)
                self.assertTrue(torch.all(tensor == 0))

    def test_torch_ones(self) -> None:
        # Test on each available device
        for device_name in self.available_devices:
            with self.subTest(device=device_name):
                set_device(device_name)
                size = (random.randint(1, 10), random.randint(1, 10))
                tensor = torch_ones(size)
                tensor_device = torch.device(
                    "cuda" if tensor.is_cuda else "mps" if tensor.is_mps else "cpu"
                )
                self.assertEqual(tensor_device, global_device())
                self.assertEqual(tensor.shape, size)
                self.assertTrue(torch.all(tensor == 1))

    def test_set_from_conf(self) -> None:
        obj = type("", (), {})()
        setattr(obj, "_config_data", {"var": 1})
        set_from_conf(obj, "variable", 0)
        self.assertTrue(hasattr(obj, "variable"))
        self.assertEqual(obj.variable, 0)  # type: ignore
        var = set_from_conf(obj, "var", 0, setter=False)
        self.assertFalse(hasattr(obj, "var"))
        self.assertEqual(var, 1)

    def test_activation_fn(self) -> None:
        self.assertIsInstance(activation_fn(None), nn.Identity)  # type: ignore
        self.assertIsInstance(activation_fn("Id"), nn.Identity)
        self.assertIsInstance(activation_fn("Softmax"), nn.Softmax)
        self.assertIsInstance(activation_fn("SELU"), nn.SELU)
        self.assertIsInstance(activation_fn("RELU"), nn.ReLU)
        with self.assertRaises(ValueError):
            activation_fn("UnknownActivation")

    def test_mini_batch_gradient_descent(self) -> None:
        # Test on each available device
        for device_name in self.available_devices:
            with self.subTest(device=device_name):
                set_device(device_name)

                callable_forward = lambda x: x**2 + 1  # noqa: E731
                cost_fn = lambda pred, y: torch.sum((pred - y) ** 2)  # noqa: E731
                x = torch.rand((5, 2), requires_grad=True, device=global_device())
                y = torch.rand((5, 1), device=global_device())
                lrate = 1e-3
                epochs = 50
                batch_size = 8

                # Test error cases
                with self.assertRaises(AttributeError):
                    mini_batch_gradient_descent(
                        callable_forward,
                        cost_fn,
                        x,
                        y,
                        lrate,
                        epochs,
                        batch_size,
                        verbose=False,
                    )
                    mini_batch_gradient_descent(
                        callable_forward,
                        cost_fn,
                        x,
                        y,
                        lrate,
                        epochs,
                        batch_size,
                        parameters=[],
                        verbose=False,
                    )

                # Test with parameters
                parameters = [x]
                mini_batch_gradient_descent(
                    callable_forward,
                    cost_fn,
                    x,
                    y,
                    lrate,
                    epochs,
                    batch_size,
                    parameters,
                    verbose=False,
                )

                # Test with model
                model = nn.Linear(2, 1, device=global_device())
                eval_fn = lambda: None  # noqa: E731
                mini_batch_gradient_descent(
                    model,
                    cost_fn,
                    x,
                    y,
                    lrate,
                    epochs,
                    batch_size,
                    eval_fn=eval_fn,
                    verbose=False,
                )

    def test_line_search(self) -> None:
        """Test line search optimization algorithm"""

        # Test simple quadratic function: f(x) = (x - 2)^2 + 1, minimum at x=2
        def quadratic_cost(x):
            return (x - 2.0) ** 2 + 1.0

        # Test basic functionality
        factor, min_loss = line_search(quadratic_cost, return_history=False)
        self.assertIsInstance(factor, float)
        self.assertIsInstance(min_loss, float)
        assert isinstance(factor, float)
        assert isinstance(min_loss, float)
        self.assertGreater(min_loss, 1.0)  # Minimum value should be close to 1
        self.assertGreater(factor, 0.0)  # Factor should be positive

        # Test with return_history=True
        factors, losses = line_search(quadratic_cost, return_history=True)
        self.assertIsInstance(factors, list)
        self.assertIsInstance(losses, list)
        assert isinstance(factors, list)
        assert isinstance(losses, list)
        self.assertEqual(len(factors), len(losses))
        self.assertGreater(len(factors), 0)

        # Test that minimum is reasonable for quadratic function
        min_idx = int(torch.argmin(torch.tensor(losses)).item())
        best_factor = factors[min_idx]
        best_loss = losses[min_idx]
        # For quadratic function, should find factor close to 2
        self.assertLess(abs(best_factor - 2.0), 1.0)  # Within reasonable range
        self.assertLess(best_loss, 2.0)  # Loss should be reasonable

    def test_batch_gradient_descent(self) -> None:
        """Test batch gradient descent implementation"""
        # Create simple optimization problem with classification setup
        target = torch.tensor([0, 1, 2])  # Classification targets

        # Simple forward function that returns a learnable parameter
        param = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True
        )

        def forward_fn():
            return param

        # CrossEntropy-like cost function
        def cost_fn(output, target):
            return torch.nn.functional.cross_entropy(output, target)

        # Create optimizer
        optimizer = torch.optim.SGD([param], lr=0.1)

        # Test basic functionality with fast=True
        loss_history, acc_history = batch_gradient_descent(
            forward_fn, cost_fn, target, optimizer, max_epochs=10, fast=True
        )

        self.assertIsInstance(loss_history, list)
        self.assertIsInstance(acc_history, list)
        self.assertEqual(len(loss_history), 10)
        self.assertGreater(len(loss_history), 0)

        # Test with fast=False and eval_fn
        eval_called = False

        def eval_fn():
            nonlocal eval_called
            eval_called = True

        param.data.fill_(0.0)  # Reset parameter
        loss_history, acc_history = batch_gradient_descent(
            forward_fn,
            cost_fn,
            target,
            optimizer,
            max_epochs=5,
            fast=False,
            eval_fn=eval_fn,
        )

        self.assertTrue(eval_called)
        self.assertEqual(len(acc_history), 5)
        # Accuracy should be between 0 and 1
        for acc in acc_history:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_calculate_true_positives(self) -> None:
        """Test classification metrics calculation"""
        # Create test data
        actual = torch.tensor([0, 0, 1, 1, 2, 2])
        predicted = torch.tensor([0, 1, 1, 1, 2, 0])

        # Test for label 0
        tp, fp, fn = calculate_true_positives(actual, predicted, 0)
        self.assertEqual(tp, 1)  # One correct prediction of label 0
        self.assertEqual(fp, 1)  # One false positive (predicted 0, actual 2)
        self.assertEqual(fn, 1)  # One false negative (predicted 1, actual 0)

        # Test for label 1
        tp, fp, fn = calculate_true_positives(actual, predicted, 1)
        self.assertEqual(tp, 2)  # Two correct predictions of label 1
        self.assertEqual(fp, 1)  # One false positive (predicted 1, actual 0)
        self.assertEqual(fn, 0)  # No false negatives

        # Test for label 2
        tp, fp, fn = calculate_true_positives(actual, predicted, 2)
        self.assertEqual(tp, 1)  # One correct prediction of label 2
        self.assertEqual(fp, 0)  # No false positives
        self.assertEqual(fn, 1)  # One false negative (predicted 0, actual 2)

    def test_f1_score_functions(self) -> None:
        """Test F1 score calculation functions"""
        # Create test data with perfect classification for label 0
        actual = torch.tensor([0, 0, 1, 1, 2, 2])
        predicted_perfect = torch.tensor([0, 0, 1, 1, 2, 2])

        # Test perfect F1 score
        f1_score = f1(actual, predicted_perfect, 0)
        self.assertAlmostEqual(f1_score, 1.0, places=5)

        # Test with some errors
        predicted_errors = torch.tensor([0, 1, 1, 1, 2, 0])
        f1_score = f1(actual, predicted_errors, 1)
        # For label 1: tp=2, fp=1, fn=0
        # precision = 2/3, recall = 2/2 = 1, f1 = 2*(2/3*1)/(2/3+1) = 4/5 = 0.8
        self.assertAlmostEqual(f1_score, 0.8, places=5)

        # Test F1 micro average
        f1_micro_score = f1_micro(actual, predicted_perfect)
        self.assertAlmostEqual(f1_micro_score, 1.0, places=5)

        # Test F1 macro average
        f1_macro_score = f1_macro(actual, predicted_perfect)
        self.assertAlmostEqual(f1_macro_score, 1.0, places=5)

        # Test with errors
        f1_micro_score = f1_micro(actual, predicted_errors)
        f1_macro_score = f1_macro(actual, predicted_errors)
        self.assertGreater(f1_micro_score, 0.0)
        self.assertLess(f1_micro_score, 1.0)
        self.assertGreater(f1_macro_score, 0.0)
        self.assertLess(f1_macro_score, 1.0)

    def test_safe_forward(self) -> None:
        """Test safe_forward function for Linear layers with various input configurations."""
        # Test on each available device
        for device_name in self.available_devices:
            with self.subTest(device=device_name):
                set_device(device_name)
                device = global_device()

                # Test normal case with non-zero features
                in_features = 5
                out_features = 3
                batch_size = 4

                # Create a mock linear layer
                linear_layer = nn.Linear(in_features, out_features, device=device)

                # Test normal forward pass
                input_tensor = torch.randn(batch_size, in_features, device=device)
                output = safe_forward(linear_layer, input_tensor)

                self.assertShapeEqual(output, (batch_size, out_features))
                # Compare with normal linear forward
                expected_output = torch.nn.functional.linear(
                    input_tensor, linear_layer.weight, linear_layer.bias
                )
                self.assertAllClose(output, expected_output)

                # Test with zero input features (edge case)
                zero_in_features = 0
                zero_linear = nn.Linear(zero_in_features, out_features, device=device)
                zero_input = torch.empty(batch_size, zero_in_features, device=device)

                zero_output = safe_forward(zero_linear, zero_input)
                self.assertShapeEqual(zero_output, (batch_size, out_features))
                # Should return zeros with requires_grad=True
                self.assertTrue(torch.all(zero_output == 0))
                self.assertTrue(zero_output.requires_grad)

                # Test input shape mismatch (should raise AssertionError)
                wrong_input = torch.randn(batch_size, in_features + 1, device=device)
                with self.assertRaises(AssertionError) as context:
                    safe_forward(linear_layer, wrong_input)

                self.assertIn("Input shape", str(context.exception))
                self.assertIn("must match the input feature size", str(context.exception))

                # Test with different batch dimensions
                input_3d = torch.randn(2, 3, in_features, device=device)
                output_3d = safe_forward(linear_layer, input_3d)
                self.assertShapeEqual(output_3d, (2, 3, out_features))

                # Test with single sample
                input_single = torch.randn(1, in_features, device=device)
                output_single = safe_forward(linear_layer, input_single)
                self.assertShapeEqual(output_single, (1, out_features))

                # Test without bias
                linear_no_bias = nn.Linear(
                    in_features, out_features, bias=False, device=device
                )
                output_no_bias = safe_forward(linear_no_bias, input_tensor)
                self.assertShapeEqual(output_no_bias, (batch_size, out_features))
                expected_no_bias = torch.nn.functional.linear(
                    input_tensor, linear_no_bias.weight, None
                )
                self.assertAllClose(output_no_bias, expected_no_bias)

    def test_compute_tensor_stats(self) -> None:
        """Test compute_tensor_stats function returns correct output types."""
        # Test with normal tensor (multiple elements)
        for tensor in [torch.randn(3, 4), torch.tensor(5.0)]:
            stats = compute_tensor_stats(tensor)

            # Check output type and keys
            self.assertIsInstance(stats, dict)
            expected_keys = {"min", "max", "mean", "std"}
            self.assertEqual(set(stats.keys()), expected_keys)

            # Check value types
            for key, value in stats.items():
                self.assertIsInstance(
                    value, float, f"Value for key '{key}' should be float"
                )

    def test_evaluate_dataset(self) -> None:
        batch_size = 4
        in_features = 5
        out_features = 2
        model = torch.nn.Linear(in_features, out_features, device=global_device())

        x = torch.rand(batch_size, in_features, device=global_device())
        y = torch.randint(0, out_features, (batch_size,), device=global_device())
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        accuracy, loss = evaluate_dataset(model, dataloader, loss_fn)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertGreaterEqual(loss, 0)
        self.assertEqual(loss, loss_fn(model(x), y).item())

        out_features = 1
        model = torch.nn.Linear(in_features, out_features, device=global_device())

        y = torch.rand(batch_size, out_features, device=global_device())
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=batch_size
        )
        loss_fn = torch.nn.MSELoss()

        accuracy, loss = evaluate_dataset(model, dataloader, loss_fn)
        self.assertEqual(accuracy, -1)
        self.assertGreaterEqual(loss, 0)
        self.assertEqual(loss, loss_fn(model(x), y).item())


if __name__ == "__main__":
    from unittest import main

    main()
