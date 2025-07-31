"""
Performance regression tests for the GroMo library.

These tests ensure that enhancements to the library don't negatively impact performance
by measuring execution time and memory usage for key operations.
"""

import time
import unittest
from typing import Dict, Any
import torch
import torch.nn as nn
import os
import json
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

from gromo.containers.growing_residual_mlp import GrowingResidualMLP
from gromo.containers.growing_mlp import GrowingMLP
from gromo.utils.utils import set_device, global_device
from tests.torch_unittest import TorchTestCase


class TestPerformanceRegression(TorchTestCase):
    """Performance regression tests for the GroMo library."""
    
    # Baseline performance thresholds (in seconds)
    BASELINE_TRAINING_TIME = {
        "GrowingResidualMLP_32_16_2": 2.0,  # seconds
        "GrowingMLP_64_2": 1.5,  # seconds
    }
    
    BASELINE_INFERENCE_TIME = {
        "GrowingResidualMLP_32_16_2": 0.05,  # seconds
        "GrowingMLP_64_2": 0.03,  # seconds
    }
    
    # Memory usage thresholds (in MB)
    BASELINE_MEMORY_USAGE = {
        "GrowingResidualMLP_32_16_2": 100,  # MB
        "GrowingMLP_64_2": 80,  # MB
    }
    
    # Performance degradation threshold (percentage)
    PERFORMANCE_DEGRADATION_THRESHOLD = 1.2  # 20% degradation allowed
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set device for testing
        set_device("cpu")
        
        # Create synthetic data for testing
        self.batch_size = 16
        self.input_shape = (3, 32, 32)
        self.num_classes = 10
        self.num_samples = 64
        
        # Create sample data
        self.x = torch.randn(self.batch_size, *self.input_shape)
        self.y_true = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Performance baseline file path
        self.baseline_file = Path("tests/performance_baseline.json")
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        else:
            # Fallback: return 0 if psutil is not available
            return 0.0
    
    def measure_time(self, func, *args, **kwargs) -> tuple[Any, float]:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    
    def create_baseline_performance_data(self) -> Dict[str, Any]:
        """Create baseline performance data for comparison."""
        return {
            "GrowingResidualMLP_32_16_2": {
                "training_time": 0.5,
                "inference_time": 0.01,
                "memory_usage": 50.0
            },
            "GrowingMLP_64_2": {
                "training_time": 0.3,
                "inference_time": 0.005,
                "memory_usage": 40.0
            }
        }
    
    def load_baseline_performance(self) -> Dict[str, Any]:
        """Load baseline performance data from file or create default."""
        if self.baseline_file.exists():
            with open(self.baseline_file, "r") as f:
                return json.load(f)
        else:
            # Create default baseline data
            baseline_data = self.create_baseline_performance_data()
            self.save_baseline_performance(baseline_data)
            return baseline_data
    
    def save_baseline_performance(self, data: Dict[str, Any]) -> None:
        """Save baseline performance data to file."""
        with open(self.baseline_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def test_growing_residual_mlp_training_performance(self):
        """Test training performance of GrowingResidualMLP."""
        # Initialize model
        model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU(),
            device=global_device()
        )
        
        # Measure memory before training
        memory_before = self.get_memory_usage()
        
        # Measure training time
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        def training_step():
            model.zero_grad()
            output = model(self.x)
            loss = self.loss_fn(output, self.y_true)
            loss.backward()
            optimizer.step()
        
        # Run multiple training steps and measure time
        training_times = []
        for _ in range(5):  # Run 5 training steps
            _, exec_time = self.measure_time(training_step)
            training_times.append(exec_time)
        
        avg_training_time = sum(training_times) / len(training_times)
        
        # Measure memory after training
        memory_after = self.get_memory_usage()
        memory_usage = memory_after - memory_before
        
        # Load baseline performance data
        baseline_data = self.load_baseline_performance()
        model_key = "GrowingResidualMLP_32_16_2"
        
        # Check if baseline exists, if not create it
        if model_key not in baseline_data:
            baseline_data[model_key] = {
                "training_time": avg_training_time,
                "memory_usage": memory_usage
            }
            self.save_baseline_performance(baseline_data)
        
        # Compare with baseline
        baseline_training_time = baseline_data[model_key]["training_time"]
        baseline_memory_usage = baseline_data[model_key]["memory_usage"]
        
        # Assert performance is within acceptable thresholds
        self.assertLessEqual(
            avg_training_time,
            baseline_training_time * self.PERFORMANCE_DEGRADATION_THRESHOLD,
            f"Training time degradation detected. Current: {avg_training_time:.4f}s, "
            f"Baseline: {baseline_training_time:.4f}s"
        )
        
        self.assertLessEqual(
            memory_usage,
            baseline_memory_usage * self.PERFORMANCE_DEGRADATION_THRESHOLD,
            f"Memory usage degradation detected. Current: {memory_usage:.2f}MB, "
            f"Baseline: {baseline_memory_usage:.2f}MB"
        )
    
    def test_growing_residual_mlp_inference_performance(self):
        """Test inference performance of GrowingResidualMLP."""
        # Initialize model
        model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU(),
            device=global_device()
        )
        
        # Warm up the model
        for _ in range(3):
            _ = model(self.x)
        
        # Measure inference time
        inference_times = []
        for _ in range(10):  # Run 10 inference steps
            _, exec_time = self.measure_time(model, self.x)
            inference_times.append(exec_time)
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        # Load baseline performance data
        baseline_data = self.load_baseline_performance()
        model_key = "GrowingResidualMLP_32_16_2"
        
        # Check if baseline exists, if not create it
        if model_key not in baseline_data:
            baseline_data[model_key] = {
                "inference_time": avg_inference_time
            }
            self.save_baseline_performance(baseline_data)
        
        # Compare with baseline
        baseline_inference_time = baseline_data[model_key].get(
            "inference_time", avg_inference_time
        )
        
        # Assert inference performance is within acceptable thresholds
        self.assertLessEqual(
            avg_inference_time,
            baseline_inference_time * self.PERFORMANCE_DEGRADATION_THRESHOLD,
            f"Inference time degradation detected. Current: {avg_inference_time:.4f}s, "
            f"Baseline: {baseline_inference_time:.4f}s"
        )
    
    def test_growing_mlp_training_performance(self):
        """Test training performance of GrowingMLP."""
        # Initialize model
        input_features = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        model = GrowingMLP(
            in_features=input_features,
            out_features=self.num_classes,
            hidden_size=64,
            number_hidden_layers=2,
            activation=nn.ReLU(),
            device=global_device()
        )
        
        # Measure memory before training
        memory_before = self.get_memory_usage()
        
        # Measure training time
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        def training_step():
            model.zero_grad()
            # Flatten input for MLP
            x_flat = self.x.view(self.x.size(0), -1)
            output = model(x_flat)
            loss = self.loss_fn(output, self.y_true)
            loss.backward()
            optimizer.step()
        
        # Run multiple training steps and measure time
        training_times = []
        for _ in range(5):  # Run 5 training steps
            _, exec_time = self.measure_time(training_step)
            training_times.append(exec_time)
        
        avg_training_time = sum(training_times) / len(training_times)
        
        # Measure memory after training
        memory_after = self.get_memory_usage()
        memory_usage = memory_after - memory_before
        
        # Load baseline performance data
        baseline_data = self.load_baseline_performance()
        model_key = "GrowingMLP_64_2"
        
        # Check if baseline exists, if not create it
        if model_key not in baseline_data:
            baseline_data[model_key] = {
                "training_time": avg_training_time,
                "memory_usage": memory_usage
            }
            self.save_baseline_performance(baseline_data)
        
        # Compare with baseline
        baseline_training_time = baseline_data[model_key]["training_time"]
        baseline_memory_usage = baseline_data[model_key]["memory_usage"]
        
        # Assert performance is within acceptable thresholds
        self.assertLessEqual(
            avg_training_time,
            baseline_training_time * self.PERFORMANCE_DEGRADATION_THRESHOLD,
            f"Training time degradation detected. Current: {avg_training_time:.4f}s, "
            f"Baseline: {baseline_training_time:.4f}s"
        )
        
        self.assertLessEqual(
            memory_usage,
            baseline_memory_usage * self.PERFORMANCE_DEGRADATION_THRESHOLD,
            f"Memory usage degradation detected. Current: {memory_usage:.2f}MB, "
            f"Baseline: {baseline_memory_usage:.2f}MB"
        )
    
    def test_growing_mlp_inference_performance(self):
        """Test inference performance of GrowingMLP."""
        # Initialize model
        input_features = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        model = GrowingMLP(
            in_features=input_features,
            out_features=self.num_classes,
            hidden_size=64,
            number_hidden_layers=2,
            activation=nn.ReLU(),
            device=global_device()
        )
        
        # Warm up the model
        x_flat = self.x.view(self.x.size(0), -1)
        for _ in range(3):
            _ = model(x_flat)
        
        # Measure inference time
        inference_times = []
        for _ in range(10):  # Run 10 inference steps
            _, exec_time = self.measure_time(model, x_flat)
            inference_times.append(exec_time)
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        # Load baseline performance data
        baseline_data = self.load_baseline_performance()
        model_key = "GrowingMLP_64_2"
        
        # Check if baseline exists, if not create it
        if model_key not in baseline_data:
            baseline_data[model_key] = {
                "inference_time": avg_inference_time
            }
            self.save_baseline_performance(baseline_data)
        
        # Compare with baseline
        baseline_inference_time = baseline_data[model_key].get(
            "inference_time", avg_inference_time
        )
        
        # Assert inference performance is within acceptable thresholds
        self.assertLessEqual(
            avg_inference_time,
            baseline_inference_time * self.PERFORMANCE_DEGRADATION_THRESHOLD,
            f"Inference time degradation detected. Current: {avg_inference_time:.4f}s, "
            f"Baseline: {baseline_inference_time:.4f}s"
        )
    
    def test_growing_functionality_performance(self):
        """Test performance of growing functionality."""
        # Initialize model
        model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU(),
            device=global_device()
        )
        
        # Measure time for initialization of computation
        _, init_time = self.measure_time(model.init_computation)
        
        # Run forward and backward pass to accumulate statistics
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        def forward_backward_pass():
            optimizer.zero_grad()
            output = model(self.x)
            loss = self.loss_fn(output, self.y_true)
            loss.backward()
            model.update_computation()
        
        # Measure time for forward/backward passes
        fb_times = []
        for _ in range(3):  # Run 3 forward/backward passes
            _, exec_time = self.measure_time(forward_backward_pass)
            fb_times.append(exec_time)
        
        avg_fb_time = sum(fb_times) / len(fb_times)
        
        # Measure time for computing optimal updates
        _, update_time = self.measure_time(model.compute_optimal_updates)
        
        # Measure time for selecting best update
        _, select_time = self.measure_time(model.select_best_update)
        
        # Measure time for applying changes
        _, apply_time = self.measure_time(model.apply_change)
        
        # Total growing time
        total_growing_time = init_time + avg_fb_time + update_time + select_time + apply_time
        
        # These operations should complete within reasonable time
        self.assertLess(
            total_growing_time,
            5.0,  # Should complete within 5 seconds
            f"Growing functionality took too long: {total_growing_time:.4f}s"
        )
    
    def test_memory_efficiency(self):
        """Test memory efficiency of the library."""
        # Skip test if psutil is not available
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available")
            
        # Measure memory before creating models
        initial_memory = self.get_memory_usage()
        
        # Create multiple models to test memory efficiency
        models = []
        for i in range(5):
            model = GrowingResidualMLP(
                in_features=self.input_shape,
                out_features=self.num_classes,
                num_features=32,
                hidden_features=16,
                num_blocks=2,
                activation=nn.ReLU(),
                device=global_device()
            )
            models.append(model)
        
        # Measure memory after creating models
        memory_after_creation = self.get_memory_usage()
        memory_per_model = (memory_after_creation - initial_memory) / len(models)
        
        # Run some operations
        for model in models:
            _ = model(self.x)
        
        # Measure memory after operations
        memory_after_operations = self.get_memory_usage()
        
        # Clean up models
        del models
        torch.cuda.empty_cache()  # Clean up GPU memory if available
        
        # Final memory measurement
        final_memory = self.get_memory_usage()
        
        # Memory should not grow uncontrollably
        self.assertLess(
            final_memory,
            initial_memory + 200,  # Should not use more than 200MB additional memory
            f"Memory usage grew too much. Initial: {initial_memory:.2f}MB, "
            f"Final: {final_memory:.2f}MB"
        )
    
    def test_batch_size_scaling_performance(self):
        """Test performance scaling with different batch sizes."""
        model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU(),
            device=global_device()
        )
        
        batch_sizes = [8, 16, 32]
        inference_times = []
        
        for batch_size in batch_sizes:
            # Create input for this batch size
            x_batch = torch.randn(batch_size, *self.input_shape)
            
            # Warm up
            for _ in range(3):
                _ = model(x_batch)
            
            # Measure inference time
            _, exec_time = self.measure_time(model, x_batch)
            inference_times.append(exec_time)
        
        # Check that inference time scales reasonably with batch size
        # We expect roughly linear scaling
        time_per_sample_8 = inference_times[0] / 8
        time_per_sample_32 = inference_times[2] / 32
        
        # Time per sample should not increase significantly with batch size
        self.assertLessEqual(
            time_per_sample_32,
            time_per_sample_8 * 1.5,  # Allow 50% increase due to overhead
            f"Inference time per sample increased too much with batch size. "
            f"Batch 8: {time_per_sample_8:.6f}s/sample, "
            f"Batch 32: {time_per_sample_32:.6f}s/sample"
        )


if __name__ == "__main__":
    unittest.main()