import random
import unittest

import torch

from gromo.utils.utils import *

from .unittest_tools import unittest_parametrize


class TestUtils(unittest.TestCase):
    def test_set_device(self) -> None:
        if torch.cuda.is_available():
            self.assertEqual(global_device(), torch.device("cuda"))
        else:
            self.assertEqual(global_device(), torch.device("cpu"))
        set_device("cuda")
        self.assertEqual(global_device(), torch.device("cuda"))
        set_device("cpu")
        self.assertEqual(global_device(), torch.device("cpu"))
        set_device(torch.device("cuda"))
        self.assertEqual(global_device(), torch.device("cuda"))
        set_device(torch.device("cpu"))
        self.assertEqual(global_device(), torch.device("cpu"))

    def test_reset_device(self) -> None:
        # Save the original device
        original_device = global_device()
        
        # Set device to something else
        if torch.cuda.is_available():
            set_device("cpu")
            self.assertEqual(global_device(), torch.device("cpu"))
        else:
            set_device("cuda")
            self.assertEqual(global_device(), torch.device("cuda"))
            
        # Reset device and check it goes back to default
        reset_device()
        if torch.cuda.is_available():
            self.assertEqual(global_device(), torch.device("cuda"))
        else:
            self.assertEqual(global_device(), torch.device("cpu"))
            
        # Restore original device
        set_device(original_device)

    def test_get_correct_device(self) -> None:
        # Create a mock object with config data
        class MockObject:
            def __init__(self):
                self._config_data = {"device": "cpu"}
        
        mock_obj = MockObject()
        
        # Test with explicit device argument
        device = get_correct_device(mock_obj, "cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(device, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Test with None (should use config data)
        device = get_correct_device(mock_obj, None)
        self.assertEqual(device, torch.device("cpu"))
        
        # Test with string device
        device = get_correct_device(mock_obj, "cpu")
        self.assertEqual(device, torch.device("cpu"))

    def test_line_search(self) -> None:
        # Create a simple convex function for testing
        # f(x) = (x - 0.5)^2 + 0.1, minimum at x=0.5
        def convex_fn(x):
            return (x - 0.5) ** 2 + 0.1
        
        # Test line search without history
        factor, min_value = line_search(convex_fn, return_history=False)
        self.assertIsInstance(factor, float)
        self.assertIsInstance(min_value, float)
        
        # Test line search with history
        factors, losses = line_search(convex_fn, return_history=True)
        self.assertIsInstance(factors, list)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(factors), len(losses))
        self.assertGreater(len(factors), 0)

    def test_batch_gradient_descent(self) -> None:
        # Create a simple model for testing
        model = nn.Linear(2, 1)
        
        # Create forward function
        def forward_fn():
            return model(torch.rand((5, 2), device=global_device()))
        
        # Create cost function
        def cost_fn(output, target):
            return torch.sum((output - target) ** 2)
        
        # Create target tensor
        target = torch.rand((5, 1), device=global_device())
        
        # Create optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Test batch gradient descent with fast=True (default)
        loss_history, acc_history = batch_gradient_descent(
            forward_fn, cost_fn, target, optimizer, max_epochs=10, fast=True
        )
        self.assertIsInstance(loss_history, list)
        self.assertIsInstance(acc_history, list)
        self.assertEqual(len(loss_history), 10)
        self.assertEqual(len(acc_history), 0)  # Empty when fast=True
        
        # Test batch gradient descent with fast=False
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Reset optimizer
        loss_history, acc_history = batch_gradient_descent(
            forward_fn, cost_fn, target, optimizer, max_epochs=5, fast=False
        )
        self.assertIsInstance(loss_history, list)
        self.assertIsInstance(acc_history, list)
        self.assertEqual(len(loss_history), 5)
        self.assertEqual(len(acc_history), 5)  # Not empty when fast=False
# Test batch_gradient_descent with eval_fn parameter
        eval_called = False
        def eval_fn():
            nonlocal eval_called
            eval_called = True
            
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Reset optimizer
        loss_history, acc_history = batch_gradient_descent(
            forward_fn, cost_fn, target, optimizer, max_epochs=5, fast=False, eval_fn=eval_fn
        )
        self.assertIsInstance(loss_history, list)
        self.assertIsInstance(acc_history, list)
        self.assertEqual(len(loss_history), 5)
        self.assertEqual(len(acc_history), 5)
        # Check that eval_fn was called
        self.assertTrue(eval_called)

    def test_calculate_true_positives(self) -> None:
        # Create sample tensors for testing
        actual = torch.tensor([0, 1, 1, 0, 2, 2, 1, 0])
        predicted = torch.tensor([0, 1, 0, 0, 2, 1, 1, 2])
        
        # Test for label 0
        tp, fp, fn = calculate_true_positives(actual, predicted, 0)
        self.assertIsInstance(tp, int)
        self.assertIsInstance(fp, int)
        self.assertIsInstance(fn, int)
        
        # Manually calculate expected values for label 0:
        # True positives: actual=0 and predicted=0 at indices [0, 3] => 2
        # False positives: actual!=0 and predicted=0 at indices [2] => 1
        # False negatives: predicted!=0 and actual=0 at indices [7] => 1
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)
        
        # Test for label 1
        tp, fp, fn = calculate_true_positives(actual, predicted, 1)
        # Manually calculate expected values for label 1:
        # True positives: actual=1 and predicted=1 at indices [1, 6] => 2
        # False positives: actual!=1 and predicted=1 at indices [5] => 1
        # False negatives: predicted!=1 and actual=1 at indices [2] => 1
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)
        
        # Test for label 2
        tp, fp, fn = calculate_true_positives(actual, predicted, 2)
        # Manually calculate expected values for label 2:
        # True positives: actual=2 and predicted=2 at indices [4] => 1
        # False positives: actual!=2 and predicted=2 at indices [7] => 1
        # False negatives: predicted!=2 and actual=2 at indices [5] => 1
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)

    def test_f1(self) -> None:
        # Create sample tensors for testing
        actual = torch.tensor([0, 1, 1, 0, 2, 2, 1, 0])
        predicted = torch.tensor([0, 1, 0, 0, 2, 1, 1, 2])
        
        # Test for label 0
        f1_score = f1(actual, predicted, 0)
        self.assertIsInstance(f1_score, float)
        
        # Manually calculate expected F1 score for label 0:
        # True positives: 2, False positives: 1, False negatives: 1
        # Precision = 2 / (2 + 1) = 0.6667
        # Recall = 2 / (2 + 1) = 0.6667
        # F1 = 2 * (0.6667 * 0.6667) / (0.6667 + 0.6667) = 2 * 0.4444 / 1.3333 = 0.6667
        self.assertAlmostEqual(f1_score, 0.6667, places=4)
# Test edge cases for f1 function
        # Case 1: Zero division in precision (tp + fp = 0)
        # This happens when there are no true positives and no false positives
        # We'll create a scenario where the label exists in actual but not in predicted
        actual_edge = torch.tensor([0, 0, 0, 0])  # Only label 0
        predicted_edge = torch.tensor([1, 1, 1, 1])  # Only label 1, different from actual
        f1_score_edge = f1(actual_edge, predicted_edge, 0)
        # With tp=0, fp=0, fn=4: precision = 0/(0+0) -> this would cause division by zero
        # The function should handle this gracefully
        self.assertIsInstance(f1_score_edge, float)
        
        # Case 2: Zero division in recall (tp + fn = 0)
        # This happens when there are no true positives and no false negatives
        # We'll create a scenario where the label doesn't exist in actual
        actual_edge2 = torch.tensor([0, 0, 0, 0])  # Only label 0
        predicted_edge2 = torch.tensor([0, 0, 0, 0])  # Only label 0
        f1_score_edge2 = f1(actual_edge2, predicted_edge2, 1)
        # Label 1 doesn't exist in actual, so tp=0, fp=0, fn=0
        # But since the label doesn't exist in actual, this is a special case
        self.assertIsInstance(f1_score_edge2, float)
        
        # Test for label 1
        f1_score = f1(actual, predicted, 1)
        # True positives: 2, False positives: 1, False negatives: 1
        # Precision = 2 / (2 + 1) = 0.6667
        # Recall = 2 / (2 + 1) = 0.6667
        # F1 = 2 * (0.6667 * 0.6667) / (0.6667 + 0.6667) = 2 * 0.4444 / 1.3333 = 0.6667
        self.assertAlmostEqual(f1_score, 0.6667, places=4)
        
        # Test for label 2
        f1_score = f1(actual, predicted, 2)
        # True positives: 1, False positives: 1, False negatives: 1
        # Precision = 1 / (1 + 1) = 0.5
        # Recall = 1 / (1 + 1) = 0.5
        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 2 * 0.25 / 1.0 = 0.5
        self.assertAlmostEqual(f1_score, 0.5, places=4)

    def test_f1_micro(self) -> None:
        # Create sample tensors for testing
        actual = torch.tensor([0, 1, 1, 0, 2, 2, 1, 0])
        predicted = torch.tensor([0, 1, 0, 0, 2, 1, 1, 2])
        
        # Test f1_micro calculation
        f1_micro_score = f1_micro(actual, predicted)
        self.assertIsInstance(f1_micro_score, float)
        
        # Manually calculate expected micro F1 score:
        # For label 0: TP=2, FP=1, FN=1
        # For label 1: TP=2, FP=1, FN=1
        # For label 2: TP=1, FP=1, FN=1
        # All TP = 2 + 2 + 1 = 5
        # All FP = 1 + 1 + 1 = 3
        # All FN = 1 + 1 + 1 = 3
        # Micro precision = 5 / (5 + 3) = 5/8 = 0.625
        # Micro recall = 5 / (5 + 3) = 5/8 = 0.625
        # Micro F1 = 2 * (0.625 * 0.625) / (0.625 + 0.625) = 2 * 0.390625 / 1.25 = 0.625
        self.assertAlmostEqual(f1_micro_score, 0.625, places=4)
        
        # Test with perfect prediction
        perfect_predicted = torch.tensor([0, 1, 1, 0, 2, 2, 1, 0])
        f1_micro_perfect = f1_micro(actual, perfect_predicted)
        # For perfect prediction: TP=8, FP=0, FN=0
        # Micro precision = 8 / (8 + 0) = 1.0
        # Micro recall = 8 / (8 + 0) = 1.0
        # Micro F1 = 2 * (1.0 * 1.0) / (1.0 + 1.0) = 1.0
        self.assertAlmostEqual(f1_micro_perfect, 1.0, places=4)

    def test_f1_macro(self) -> None:
        # Create sample tensors for testing
        actual = torch.tensor([0, 1, 1, 0, 2, 2, 1, 0])
        predicted = torch.tensor([0, 1, 0, 0, 2, 1, 1, 2])
        
        # Test f1_macro calculation
        f1_macro_score = f1_macro(actual, predicted)
        self.assertIsInstance(f1_macro_score, float)
        
        # Manually calculate expected macro F1 score:
        # We already calculated individual F1 scores in test_f1:
        # F1 for label 0: 0.6667
        # F1 for label 1: 0.6667
        # F1 for label 2: 0.5
        # Macro F1 = (0.6667 + 0.6667 + 0.5) / 3 = 1.8334 / 3 = 0.6111
        self.assertAlmostEqual(f1_macro_score, 0.6111, places=4)
        
        # Test with perfect prediction
        perfect_predicted = torch.tensor([0, 1, 1, 0, 2, 2, 1, 0])
        f1_macro_perfect = f1_macro(actual, perfect_predicted)
        # With perfect prediction, all F1 scores should be 1.0
        # Macro F1 = (1.0 + 1.0 + 1.0) / 3 = 1.0
        self.assertAlmostEqual(f1_macro_perfect, 1.0, places=4)

    def test_torch_zeros(self) -> None:
        size = (random.randint(1, 10), random.randint(1, 10))
        tensor = torch_zeros(size)
        tensor_device = torch.device("cuda" if tensor.is_cuda else "cpu")
        self.assertEqual(tensor_device, global_device())
        self.assertEqual(tensor.shape, size)
        self.assertTrue(torch.all(tensor == 0))

    def test_torch_ones(self) -> None:
        size = (random.randint(1, 10), random.randint(1, 10))
        tensor = torch_ones(size)
        tensor_device = torch.device("cuda" if tensor.is_cuda else "cpu")
        self.assertEqual(tensor_device, global_device())
        self.assertEqual(tensor.shape, size)
        self.assertTrue(torch.all(tensor == 1))

    def test_set_from_conf(self) -> None:
        obj = type("", (), {})()
        setattr(obj, "_config_data", {"var": 1})
        set_from_conf(obj, "variable", 0)
        self.assertTrue(hasattr(obj, "variable"))
        self.assertEqual(obj.variable, 0)
        var = set_from_conf(obj, "var", 0, setter=False)
        self.assertFalse(hasattr(obj, "var"))
        self.assertEqual(var, 1)

    def test_activation_fn(self) -> None:
        self.assertIsInstance(activation_fn(None), nn.Identity)
        self.assertIsInstance(activation_fn("Id"), nn.Identity)
        self.assertIsInstance(activation_fn("Test"), nn.Identity)
        self.assertIsInstance(activation_fn("Softmax"), nn.Softmax)
        self.assertIsInstance(activation_fn("SELU"), nn.SELU)
        self.assertIsInstance(activation_fn("RELU"), nn.ReLU)

    def test_mini_batch_gradient_descent(self) -> None:
        callable_forward = lambda x: x**2 + 1
        cost_fn = lambda pred, y: torch.sum((pred - y) ** 2)
        x = torch.rand((5, 2), requires_grad=True, device=global_device())
        y = torch.rand((5, 1), device=global_device())
        lrate = 1e-3
        epochs = 50
        batch_size = 8
        with self.assertRaises(AttributeError):
            mini_batch_gradient_descent(
                callable_forward, cost_fn, x, y, lrate, epochs, batch_size, verbose=False
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

        # Test with verbose=True to cover print statements
        model = nn.Linear(2, 1, device=global_device())
        mini_batch_gradient_descent(
            model,
            cost_fn,
            x,
            y,
            lrate,
            max_epochs=20,  # Increase epochs to ensure epoch % 10 == 0
            batch_size=batch_size,
            verbose=True,
        )

        model = nn.Linear(2, 1, device=global_device())
        eval_fn = lambda: None
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

    def test_mini_batch_gradient_descent_with_nn_module(self):
        """
        Test mini_batch_gradient_descent when model is an nn.Module.
        This covers lines 275-276 in utils.py.
        """
        # Create a simple model
        model = nn.Linear(2, 1, device=global_device())
        
        # Create sample data
        x = torch.rand((10, 2), device=global_device())
        y = torch.rand((10, 1), device=global_device())
        
        # Define cost function
        cost_fn = lambda pred, y: torch.sum((pred - y) ** 2)
        
        # Test with model as nn.Module (should use model.parameters())
        loss_history, acc_history = mini_batch_gradient_descent(
            model,
            cost_fn,
            x,
            y,
            lrate=1e-3,
            max_epochs=5,
            batch_size=4,
            verbose=False,
        )
        
        # Check that we get valid results
        self.assertIsInstance(loss_history, list)
        self.assertIsInstance(acc_history, list)
        self.assertEqual(len(loss_history), 5)

    def test_mini_batch_gradient_descent_gradient_norm_calculation(self):
        """
        Test gradient norm calculation in mini_batch_gradient_descent.
        This covers lines 296-300 in utils.py.
        """
        # Create a simple model
        model = nn.Linear(2, 1, device=global_device())
        
        # Create sample data
        x = torch.rand((10, 2), device=global_device())
        y = torch.rand((10, 1), device=global_device())
        
        # Define cost function
        cost_fn = lambda pred, y: torch.sum((pred - y) ** 2)
        
        # Test with fast=False to trigger gradient norm calculation
        loss_history, acc_history = mini_batch_gradient_descent(
            model,
            cost_fn,
            x,
            y,
            lrate=1e-3,
            max_epochs=3,
            batch_size=5,
            fast=False,  # This will trigger the gradient norm calculation
            verbose=False,
        )
        
        # Check that we get valid results
        self.assertIsInstance(loss_history, list)
        self.assertIsInstance(acc_history, list)

    def test_mini_batch_gradient_descent_with_eval_fn(self):
        """
        Test mini_batch_gradient_descent with eval_fn parameter.
        This covers line 308 in utils.py.
        """
        # Create a simple model
        model = nn.Linear(2, 1, device=global_device())
        
        # Create sample data
        x = torch.rand((10, 2), device=global_device())
        y = torch.rand((10, 1), device=global_device())
        
        # Define cost function
        cost_fn = lambda pred, y: torch.sum((pred - y) ** 2)
        
        # Create eval function that tracks if it was called
        eval_calls = []
        def eval_fn():
            eval_calls.append(True)
        
        # Test with eval_fn parameter
        loss_history, acc_history = mini_batch_gradient_descent(
            model,
            cost_fn,
            x,
            y,
            lrate=1e-3,
            max_epochs=3,
            batch_size=5,
            eval_fn=eval_fn,
            fast=False,
            verbose=False,
        )
        
        # Check that we get valid results
        self.assertIsInstance(loss_history, list)
        self.assertIsInstance(acc_history, list)
        
        # Check that eval_fn was called
        self.assertGreater(len(eval_calls), 0)

    def test_mini_batch_gradient_descent_verbose_output(self):
        """
        Test mini_batch_gradient_descent with verbose=True.
        This covers lines 311-314 in utils.py.
        """
        # Create a simple model
        model = nn.Linear(2, 1, device=global_device())
        
        # Create sample data
        x = torch.rand((10, 2), device=global_device())
        y = torch.rand((10, 1), device=global_device())
        
        # Define cost function
        cost_fn = lambda pred, y: torch.sum((pred - y) ** 2)
        
        # Test with verbose=True and epochs that are multiples of 10
        # to trigger the verbose print statements
        loss_history, acc_history = mini_batch_gradient_descent(
            model,
            cost_fn,
            x,
            y,
            lrate=1e-3,
            max_epochs=20,  # This will trigger print at epochs 0, 10
            batch_size=5,
            fast=False,
            verbose=True,  # This will trigger the verbose print statements
        )
        
        # Check that we get valid results
        self.assertIsInstance(loss_history, list)
        self.assertIsInstance(acc_history, list)

    def test_line_search(self):
        """
        Test the line_search function.
        This covers line 211 in utils.py.
        """
        # Create a simple convex function for testing
        # f(x) = (x - 0.5)^2 + 0.1, minimum at x=0.5
        def convex_fn(x):
            return (x - 0.5) ** 2 + 0.1
        
        # Test line search without history
        factor, min_value = line_search(convex_fn, return_history=False)
        self.assertIsInstance(factor, float)
        self.assertIsInstance(min_value, float)
        
        # Test line search with history
        factors, losses = line_search(convex_fn, return_history=True)
        self.assertIsInstance(factors, list)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(factors), len(losses))
        self.assertGreater(len(factors), 0)

    def test_batch_gradient_descent(self):
        """
        Test the batch_gradient_descent function.
        This covers lines 360-388 in utils.py.
        """
        # Create a simple model for testing
        model = nn.Linear(2, 1)
        
        # Create forward function
        def forward_fn():
            return model(torch.rand((5, 2), device=global_device()))
        
        # Create cost function
        def cost_fn(output, target):
            return torch.sum((output - target) ** 2)
        
        # Create target tensor
        target = torch.rand((5, 1), device=global_device())
        
        # Create optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Test batch gradient descent with fast=True (default)
        loss_history, acc_history = batch_gradient_descent(
            forward_fn, cost_fn, target, optimizer, max_epochs=10, fast=True
        )
        self.assertIsInstance(loss_history, list)
        self.assertIsInstance(acc_history, list)
        self.assertEqual(len(loss_history), 10)
        self.assertEqual(len(acc_history), 0)  # Empty when fast=True
        
        # Test batch gradient descent with fast=False
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Reset optimizer
        loss_history, acc_history = batch_gradient_descent(
            forward_fn, cost_fn, target, optimizer, max_epochs=5, fast=False
        )
        self.assertIsInstance(loss_history, list)
        self.assertIsInstance(acc_history, list)
        self.assertEqual(len(loss_history), 5)
        self.assertEqual(len(acc_history), 5)  # Not empty when fast=False

        # Test batch_gradient_descent with eval_fn parameter
        eval_called = False
        def eval_fn():
            nonlocal eval_called
            eval_called = True
            
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Reset optimizer
        loss_history, acc_history = batch_gradient_descent(
            forward_fn, cost_fn, target, optimizer, max_epochs=5, fast=False, eval_fn=eval_fn
        )
        self.assertIsInstance(loss_history, list)
        self.assertIsInstance(acc_history, list)
        self.assertEqual(len(loss_history), 5)
        self.assertEqual(len(acc_history), 5)
        # Check that eval_fn was called
        self.assertTrue(eval_called)

    def test_f1_edge_cases(self):
        """
        Test edge cases for the f1 function.
        This covers lines 435-439 in utils.py.
        """
        # Create sample tensors for testing
        actual = torch.tensor([0, 1, 1, 0, 2, 2, 1, 0])
        predicted = torch.tensor([0, 1, 0, 0, 2, 1, 1, 2])
        
        # Test for label 0
        f1_score = f1(actual, predicted, 0)
        self.assertIsInstance(f1_score, float)
        
        # Test edge cases for f1 function
        # Case 1: Zero division in precision (tp + fp = 0)
        # This happens when there are no true positives and no false positives
        # We'll create a scenario where the label exists in actual but not in predicted
        actual_edge = torch.tensor([0, 0, 0, 0])  # Only label 0
        predicted_edge = torch.tensor([1, 1, 1, 1])  # Only label 1, different from actual
        f1_score_edge = f1(actual_edge, predicted_edge, 0)
        # With tp=0, fp=0, fn=4: precision = 0/(0+0) -> this would cause division by zero
        # The function should handle this gracefully
        self.assertIsInstance(f1_score_edge, float)
        
        # Case 2: Zero division in recall (tp + fn = 0)
        # This happens when there are no true positives and no false negatives
        # We'll create a scenario where the label doesn't exist in actual
        actual_edge2 = torch.tensor([0, 0, 0, 0])  # Only label 0
        predicted_edge2 = torch.tensor([0, 0, 0, 0])  # Only label 0
        f1_score_edge2 = f1(actual_edge2, predicted_edge2, 1)
        # Label 1 doesn't exist in actual, so tp=0, fp=0, fn=0
        # But since the label doesn't exist in actual, this is a special case
        self.assertIsInstance(f1_score_edge2, float)

    def test_f1_micro_edge_cases(self):
        """
        Test edge cases for the f1_micro function.
        This covers line 491 in utils.py.
        """
        # Create sample tensors for testing
        actual = torch.tensor([0, 1, 1, 0, 2, 2, 1, 0])
        predicted = torch.tensor([0, 1, 0, 0, 2, 1, 1, 2])
        
        # Test f1_micro calculation
        f1_micro_score = f1_micro(actual, predicted)
        self.assertIsInstance(f1_micro_score, float)
        
        # Test with perfect prediction
        perfect_predicted = torch.tensor([0, 1, 1, 0, 2, 2, 1, 0])
        f1_micro_perfect = f1_micro(actual, perfect_predicted)
        # For perfect prediction: TP=8, FP=0, FN=0
        # Micro precision = 8 / (8 + 0) = 1.0
        # Micro recall = 8 / (8 + 0) = 1.0
        # Micro F1 = 2 * (1.0 * 1.0) / (1.0 + 1.0) = 1.0
        self.assertAlmostEqual(f1_micro_perfect, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
