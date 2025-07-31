import unittest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

from gromo.containers.growing_residual_mlp import GrowingResidualMLP, GrowingResidualBlock
from gromo.modules.linear_growing_module import LinearGrowingModule, LinearMergeGrowingModule
from gromo.containers.growing_mlp import GrowingMLP
from gromo.containers.growing_graph_network import GrowingGraphNetwork
from gromo.config.loader import load_config
from gromo.utils.utils import set_device, global_device


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow of the GroMo library."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set device for testing
        set_device("cpu")
        
        # Create synthetic data for testing
        self.batch_size = 8
        self.input_shape = (3, 32, 32)
        self.num_classes = 10
        self.num_samples = 32
        
        # Create sample data
        self.x = torch.randn(self.batch_size, *self.input_shape)
        self.y_true = torch.randint(0, self.num_classes, (self.batch_size,))
        
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
    def test_end_to_end_residual_mlp_workflow(self):
        """Test complete end-to-end workflow with GrowingResidualMLP."""
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
        
        # Test initial forward pass
        output = model(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
        # Test configuration loading
        config, method = load_config()
        self.assertIsInstance(config, dict)
        
        # Test initialization of computation for growth
        model.init_computation()
        self.assertTrue(all(layer.tensor_s.samples == 0 for layer in model._growing_layers))
        
        # Test forward pass with data accumulation
        output = model(self.x)
        loss = self.loss_fn(output, self.y_true)
        loss.backward()
        
        # Test update computation
        model.update_computation()
        self.assertTrue(all(layer.tensor_s.samples > 0 for layer in model._growing_layers))
        
        # Test optimal updates computation
        model.compute_optimal_updates()
        for layer in model._growing_layers:
            self.assertIsNotNone(layer.optimal_delta_layer)
            self.assertIsNotNone(layer.parameter_update_decrease)
        
        # Test selecting best update
        model.select_best_update()
        self.assertIsNotNone(model.currently_updated_layer_index)
        
        # Test applying changes
        initial_params = sum(p.numel() for p in model.parameters())
        model.apply_change()
        final_params = sum(p.numel() for p in model.parameters())
        
        # After applying changes, the currently updated layer index should be reset to None
        self.assertIsNone(model.currently_updated_layer_index)
        
        # Clean up
        model.reset_computation()
        
    def test_data_flow_between_modules(self):
        """Test data flow between different modules in a custom network."""
        # Create a simple network with multiple growing modules
        input_features = 64
        hidden_features = 32
        output_features = 10
        
        # Create modules
        layer1 = LinearGrowingModule(
            input_features, 
            hidden_features, 
            name="layer1"
        )
        layer2 = LinearGrowingModule(
            hidden_features, 
            output_features, 
            previous_module=layer1,
            name="layer2"
        )
        
        # Connect modules
        layer1.next_module = layer2
        
        # Enable input storage for layer2 to access its input
        layer2.store_input = True
        
        # Test forward pass
        x = torch.randn(16, input_features)
        y1 = layer1(x)
        y2 = layer2(y1)
        
        self.assertEqual(y1.shape, (16, hidden_features))
        self.assertEqual(y2.shape, (16, output_features))
        
        # Test that data flows correctly between modules
        self.assertTrue(torch.allclose(layer2.input, y1))
        
        # Test extended forward
        y1_ext, y2_ext = layer1.extended_forward(x)
        y2_ext_final, _ = layer2.extended_forward(y1_ext, y2_ext)
        
        self.assertEqual(y1_ext.shape, y1.shape)
        self.assertEqual(y2_ext_final.shape, y2.shape)
        
    def test_configuration_initialization(self):
        """Test configuration and initialization processes."""
        # Test loading configuration
        config, method = load_config()
        self.assertIsInstance(config, dict)
        
        # Test with example config file if it exists
        example_config_path = Path("example_gromo.config")
        if example_config_path.exists():
            # Temporarily set environment to use example config
            original_cwd = os.getcwd()
            try:
                os.chdir(example_config_path.parent)
                config, method = load_config()
                self.assertIsInstance(config, dict)
                self.assertIn(method, ["gromo.config", "pyproject.toml"])
            finally:
                os.chdir(original_cwd)
        
        # Test model initialization with configuration
        model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU()
        )
        
        # Verify model was initialized with correct device
        self.assertEqual(model.device, global_device())
        
    def test_growing_functionality_across_containers(self):
        """Test growing functionality across different container types."""
        # Test GrowingResidualMLP
        residual_model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU()
        )
        
        # Test GrowingMLP
        mlp_model = GrowingMLP(
            in_features=self.input_shape[0] * self.input_shape[1] * self.input_shape[2],
            out_features=self.num_classes,
            hidden_size=32,
            number_hidden_layers=2,
            activation=nn.ReLU()
        )
        
        # Verify both models have growing layers
        self.assertGreater(len(residual_model._growing_layers), 0)
        self.assertGreater(len(mlp_model._growing_layers), 0)
        
        # Test initialization of growing layers
        residual_model.set_growing_layers()
        mlp_model.set_growing_layers()
        self.assertIsNotNone(residual_model._growing_layers)
        self.assertIsNotNone(mlp_model._growing_layers)
        
        # Test growing layer properties
        for layer in residual_model._growing_layers:
            self.assertTrue(hasattr(layer, 'compute_optimal_updates'))
            self.assertTrue(hasattr(layer, 'apply_change'))
            
        for layer in mlp_model._growing_layers:
            self.assertTrue(hasattr(layer, 'compute_optimal_updates'))
            self.assertTrue(hasattr(layer, 'apply_change'))
            
    def test_error_handling_in_integrated_scenarios(self):
        """Test error handling in integrated scenarios."""
        model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU()
        )
        
        # Test error when applying change without selecting update
        with self.assertRaises(AssertionError):
            model.currently_updated_layer_index = None
            model.apply_change()
            
        # Test error when computing updates without initialization
        model.reset_computation()
        with self.assertRaises(AssertionError):
            model.select_best_update()
            
    def test_complete_growth_cycle(self):
        """Test a complete growth cycle from initialization to application."""
        # Initialize model
        model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU()
        )
        
        # Store initial state
        initial_param_count = model.number_of_parameters()
        initial_output = model(self.x)
        
        # Initialize computation
        model.init_computation()
        
        # Run multiple batches to accumulate statistics
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for i in range(3):
            optimizer.zero_grad()
            output = model(self.x)
            loss = self.loss_fn(output, self.y_true)
            loss.backward()
            model.update_computation()
            
        # Compute optimal updates
        model.compute_optimal_updates()
        
        # Select and apply best update
        model.select_best_update()
        model.apply_change()
        
        # Verify changes were applied
        final_param_count = model.number_of_parameters()
        self.assertGreaterEqual(final_param_count, initial_param_count)
        
        # Verify model still works after growth
        final_output = model(self.x)
        self.assertEqual(final_output.shape, (self.batch_size, self.num_classes))
        
    def test_cross_container_data_consistency(self):
        """Test data consistency when using different container types."""
        # Create two different model types with same configuration
        input_features = 128
        out_features = 10
        
        # Test simple MLP
        mlp_model = GrowingMLP(
            in_features=input_features,
            out_features=out_features,
            hidden_size=64,
            number_hidden_layers=2,
            activation=nn.ReLU()
        )
        
        # Test residual MLP
        residual_model = GrowingResidualMLP(
            in_features=(input_features,),
            out_features=out_features,
            num_features=64,
            hidden_features=32,
            num_blocks=2,
            activation=nn.ReLU()
        )
        
        # Create test data
        test_input = torch.randn(4, input_features)
        
        # Both should produce valid outputs
        mlp_output = mlp_model(test_input)
        residual_output = residual_model(test_input)
        
        self.assertEqual(mlp_output.shape, (4, out_features))
        self.assertEqual(residual_output.shape, (4, out_features))
        
        # Test that both models can be initialized and configured similarly
        self.assertEqual(mlp_model.device, residual_model.device)
        
    def test_module_connection_and_flow(self):
        """Test proper connection and data flow between modules."""
        # Create a sequence of modules
        input_size = 16
        hidden_sizes = [32, 64, 32]
        output_size = 10
        
        # Create modules with proper connections
        layers = []
        prev_layer = None
        
        # Input layer
        layer1 = LinearGrowingModule(input_size, hidden_sizes[0], name="input_layer")
        layers.append(layer1)
        prev_layer = layer1
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes[1:], 1):
            layer = LinearGrowingModule(
                prev_layer.out_features, 
                hidden_size, 
                previous_module=prev_layer,
                name=f"hidden_layer_{i}"
            )
            prev_layer.next_module = layer
            layers.append(layer)
            prev_layer = layer
            
        # Output layer
        output_layer = LinearGrowingModule(
            prev_layer.out_features, 
            output_size, 
            previous_module=prev_layer,
            name="output_layer"
        )
        prev_layer.next_module = output_layer
        layers.append(output_layer)
        
        # Test forward pass through connected modules
        x = torch.randn(8, input_size)
        current_x = x
        
        # Forward through each layer
        for layer in layers:
            current_x = layer(current_x)
            
        # Final output should have correct shape
        self.assertEqual(current_x.shape, (8, output_size))
        
        # Test that modules are properly connected
        for i in range(len(layers) - 1):
            self.assertEqual(layers[i].next_module, layers[i + 1])
            self.assertEqual(layers[i + 1].previous_module, layers[i])
            
    def test_statistics_and_updates_flow(self):
        """Test that statistics and updates flow correctly through connected modules."""
        # Create a simple two-layer network
        layer1 = LinearGrowingModule(16, 32, name="layer1")
        layer2 = LinearGrowingModule(32, 10, previous_module=layer1, name="layer2")
        layer1.next_module = layer2
        
        # Initialize computation
        layer1.init_computation()
        layer2.init_computation()
        
        # Forward and backward pass
        x = torch.randn(8, 16)
        y = layer2(layer1(x))
        loss = torch.norm(y)
        loss.backward()
        
        # Update computation
        layer1.update_computation()
        layer2.update_computation()
        
        # Verify statistics were computed
        self.assertGreater(layer1.tensor_s().sum(), 0)
        self.assertGreater(layer2.tensor_s().sum(), 0)
        
        # Compute optimal updates
        layer1.compute_optimal_updates()
        layer2.compute_optimal_updates()
        
        # Verify updates were computed
        self.assertIsNotNone(layer1.optimal_delta_layer)
        self.assertIsNotNone(layer2.optimal_delta_layer)
        
        # Test applying changes
        initial_params1 = sum(p.numel() for p in layer1.parameters())
        initial_params2 = sum(p.numel() for p in layer2.parameters())
        
        # Apply changes to both layers
        layer1.apply_change()
        layer2.apply_change()
        
        # Verify parameters were updated
        final_params1 = sum(p.numel() for p in layer1.parameters())
        final_params2 = sum(p.numel() for p in layer2.parameters())
        
        # Clean up
        layer1.delete_update()
        layer2.delete_update()
        
    def test_end_to_end_mlp_workflow(self):
        """Test complete end-to-end workflow with GrowingMLP."""
        # Initialize model
        model = GrowingMLP(
            in_features=self.input_shape[0] * self.input_shape[1] * self.input_shape[2],
            out_features=self.num_classes,
            hidden_size=32,
            number_hidden_layers=2,
            activation=nn.ReLU()
        )
        
        # Test initial forward pass
        x_flat = self.x.view(self.batch_size, -1)
        output = model(x_flat)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
        # Test configuration loading
        config, method = load_config()
        self.assertIsInstance(config, dict)
        
        # Test initialization of computation for growth
        model.init_computation()
        self.assertTrue(all(layer.tensor_s.samples == 0 for layer in model._growing_layers))
        
        # Test forward pass with data accumulation
        output = model(x_flat)
        loss = self.loss_fn(output, self.y_true)
        loss.backward()
        
        # Test update computation
        model.update_computation()
        self.assertTrue(all(layer.tensor_s.samples > 0 for layer in model._growing_layers))
        
        # Test optimal updates computation
        model.compute_optimal_updates()
        for layer in model._growing_layers:
            self.assertIsNotNone(layer.optimal_delta_layer)
            self.assertIsNotNone(layer.parameter_update_decrease)
        
        # Test selecting best update
        model.select_best_update()
        self.assertIsNotNone(model.currently_updated_layer_index)
        
        # Test applying changes
        initial_params = sum(p.numel() for p in model.parameters())
        model.apply_change()
        final_params = sum(p.numel() for p in model.parameters())
        
        # After applying changes, the currently updated layer index should be reset to None
        self.assertIsNone(model.currently_updated_layer_index)
        
        # Clean up
        model.reset_computation()
        
    def test_data_flow_between_different_containers(self):
        """Test data flow between different container types in a complex network."""
        # Create a simple network with multiple growing modules from different containers
        input_features = 64
        hidden_features = 32
        output_features = 10
        
        # Create modules from different container types
        mlp_layer = LinearGrowingModule(
            input_features,
            hidden_features,
            name="mlp_layer"
        )
        
        residual_layer = LinearGrowingModule(
            hidden_features,
            output_features,
            previous_module=mlp_layer,
            name="residual_layer"
        )
        
        # Connect modules
        mlp_layer.next_module = residual_layer
        
        # Enable input storage for residual_layer to access its input
        residual_layer.store_input = True
        
        # Test forward pass
        x = torch.randn(16, input_features)
        y1 = mlp_layer(x)
        y2 = residual_layer(y1)
        
        self.assertEqual(y1.shape, (16, hidden_features))
        self.assertEqual(y2.shape, (16, output_features))
        
        # Test that data flows correctly between modules
        self.assertTrue(torch.allclose(residual_layer.input, y1))
        
        # Test extended forward
        y1_ext, y2_ext = mlp_layer.extended_forward(x)
        y2_ext_final, _ = residual_layer.extended_forward(y1_ext, y2_ext)
        
        self.assertEqual(y1_ext.shape, y1.shape)
        self.assertEqual(y2_ext_final.shape, y2.shape)
        
    def test_configuration_and_initialization_processes(self):
        """Test configuration and initialization processes across different components."""
        # Test loading configuration
        config, method = load_config()
        self.assertIsInstance(config, dict)
        
        # Test with example config file if it exists
        example_config_path = Path("example_gromo.config")
        if example_config_path.exists():
            # Temporarily set environment to use example config
            original_cwd = os.getcwd()
            try:
                os.chdir(example_config_path.parent)
                config, method = load_config()
                self.assertIsInstance(config, dict)
                self.assertIn(method, ["gromo.config", "pyproject.toml"])
            finally:
                os.chdir(original_cwd)
        
        # Test model initialization with configuration
        model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU()
        )
        
        # Verify model was initialized with correct device
        self.assertEqual(model.device, global_device())
        
        # Test initialization of different container types
        mlp_model = GrowingMLP(
            in_features=self.input_shape[0] * self.input_shape[1] * self.input_shape[2],
            out_features=self.num_classes,
            hidden_size=32,
            number_hidden_layers=2,
            activation=nn.ReLU()
        )
        
        # Verify both models were initialized correctly
        self.assertIsNotNone(model)
        self.assertIsNotNone(mlp_model)
        
    def test_growing_functionality_across_container_types(self):
        """Test growing functionality across different container types."""
        # Test GrowingResidualMLP
        residual_model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU()
        )
        
        # Test GrowingMLP
        mlp_model = GrowingMLP(
            in_features=self.input_shape[0] * self.input_shape[1] * self.input_shape[2],
            out_features=self.num_classes,
            hidden_size=32,
            number_hidden_layers=2,
            activation=nn.ReLU()
        )
        
        # Verify both models have growing layers
        self.assertGreater(len(residual_model._growing_layers), 0)
        self.assertGreater(len(mlp_model._growing_layers), 0)
        
        # Test initialization of growing layers
        residual_model.set_growing_layers()
        mlp_model.set_growing_layers()
        self.assertIsNotNone(residual_model._growing_layers)
        self.assertIsNotNone(mlp_model._growing_layers)
        
        # Test growing layer properties
        for layer in residual_model._growing_layers:
            self.assertTrue(hasattr(layer, 'compute_optimal_updates'))
            self.assertTrue(hasattr(layer, 'apply_change'))
            
        for layer in mlp_model._growing_layers:
            self.assertTrue(hasattr(layer, 'compute_optimal_updates'))
            self.assertTrue(hasattr(layer, 'apply_change'))
            
        # Test complete growth cycle for both models
        for model in [residual_model, mlp_model]:
            # Store initial state
            initial_param_count = model.number_of_parameters()
            if isinstance(model, GrowingResidualMLP):
                initial_output = model(self.x)
            else:
                x_flat = self.x.view(self.batch_size, -1)
                initial_output = model(x_flat)
            
            # Initialize computation
            model.init_computation()
            
            # Run multiple batches to accumulate statistics
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for i in range(3):
                optimizer.zero_grad()
                if isinstance(model, GrowingResidualMLP):
                    output = model(self.x)
                else:
                    x_flat = self.x.view(self.batch_size, -1)
                    output = model(x_flat)
                loss = self.loss_fn(output, self.y_true)
                loss.backward()
                model.update_computation()
                
            # Compute optimal updates
            model.compute_optimal_updates()
            
            # Select and apply best update
            model.select_best_update()
            model.apply_change()
            
            # Verify changes were applied
            final_param_count = model.number_of_parameters()
            self.assertGreaterEqual(final_param_count, initial_param_count)
            
            # Verify model still works after growth
            if isinstance(model, GrowingResidualMLP):
                final_output = model(self.x)
            else:
                x_flat = self.x.view(self.batch_size, -1)
                final_output = model(x_flat)
            self.assertEqual(final_output.shape, (self.batch_size, self.num_classes))
        
    def test_error_handling_in_integrated_scenarios(self):
        """Test error handling in integrated scenarios across different components."""
        # Test with GrowingResidualMLP
        residual_model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU()
        )
        
        # Test with GrowingMLP
        mlp_model = GrowingMLP(
            in_features=self.input_shape[0] * self.input_shape[1] * self.input_shape[2],
            out_features=self.num_classes,
            hidden_size=32,
            number_hidden_layers=2,
            activation=nn.ReLU()
        )
        
        for model in [residual_model, mlp_model]:
            # Test error when applying change without selecting update
            with self.assertRaises(AssertionError):
                model.currently_updated_layer_index = None
                model.apply_change()
                
            # Test error when computing updates without initialization
            model.reset_computation()
            with self.assertRaises(AssertionError):
                model.select_best_update()
                
            # Test error when no layer to update
            with self.assertRaises(AssertionError):
                _ = model.currently_updated_layer
                
    def test_cross_container_data_consistency(self):
        """Test data consistency when using different container types."""
        # Create two different model types with same configuration
        input_features = 128
        out_features = 10
        
        # Test simple MLP
        mlp_model = GrowingMLP(
            in_features=input_features,
            out_features=out_features,
            hidden_size=64,
            number_hidden_layers=2,
            activation=nn.ReLU()
        )
        
        # Test residual MLP
        residual_model = GrowingResidualMLP(
            in_features=(input_features,),
            out_features=out_features,
            num_features=64,
            hidden_features=32,
            num_blocks=2,
            activation=nn.ReLU()
        )
        
        # Create test data
        test_input = torch.randn(4, input_features)
        
        # Both should produce valid outputs
        mlp_output = mlp_model(test_input)
        residual_output = residual_model(test_input)
        
        self.assertEqual(mlp_output.shape, (4, out_features))
        self.assertEqual(residual_output.shape, (4, out_features))
        
        # Test that both models can be initialized and configured similarly
        self.assertEqual(mlp_model.device, residual_model.device)
        
    def test_module_connection_and_flow(self):
        """Test proper connection and data flow between modules."""
        # Create a sequence of modules
        input_size = 16
        hidden_sizes = [32, 64, 32]
        output_size = 10
        
        # Create modules with proper connections
        layers = []
        prev_layer = None
        
        # Input layer
        layer1 = LinearGrowingModule(input_size, hidden_sizes[0], name="input_layer")
        layers.append(layer1)
        prev_layer = layer1
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes[1:], 1):
            layer = LinearGrowingModule(
                prev_layer.out_features,
                hidden_size,
                previous_module=prev_layer,
                name=f"hidden_layer_{i}"
            )
            prev_layer.next_module = layer
            layers.append(layer)
            prev_layer = layer
            
        # Output layer
        output_layer = LinearGrowingModule(
            prev_layer.out_features,
            output_size,
            previous_module=prev_layer,
            name="output_layer"
        )
        prev_layer.next_module = output_layer
        layers.append(output_layer)
        
        # Test forward pass through connected modules
        x = torch.randn(8, input_size)
        current_x = x
        
        # Forward through each layer
        for layer in layers:
            current_x = layer(current_x)
            
        # Final output should have correct shape
        self.assertEqual(current_x.shape, (8, output_size))
        
        # Test that modules are properly connected
        for i in range(len(layers) - 1):
            self.assertEqual(layers[i].next_module, layers[i + 1])
            self.assertEqual(layers[i + 1].previous_module, layers[i])
            
    def test_statistics_and_updates_flow(self):
        """Test that statistics and updates flow correctly through connected modules."""
        # Create a simple two-layer network
        layer1 = LinearGrowingModule(16, 32, name="layer1")
        layer2 = LinearGrowingModule(32, 10, previous_module=layer1, name="layer2")
        layer1.next_module = layer2
        
        # Initialize computation
        layer1.init_computation()
        layer2.init_computation()
        
        # Forward and backward pass
        x = torch.randn(8, 16)
        y = layer2(layer1(x))
        loss = torch.norm(y)
        loss.backward()
        
        # Update computation
        layer1.update_computation()
        layer2.update_computation()
        
        # Verify statistics were computed
        self.assertGreater(layer1.tensor_s().sum(), 0)
        self.assertGreater(layer2.tensor_s().sum(), 0)
        
        # Compute optimal updates
        layer1.compute_optimal_updates()
        layer2.compute_optimal_updates()
        
        # Verify updates were computed
        self.assertIsNotNone(layer1.optimal_delta_layer)
        self.assertIsNotNone(layer2.optimal_delta_layer)
        
        # Test applying changes
        initial_params1 = sum(p.numel() for p in layer1.parameters())
        initial_params2 = sum(p.numel() for p in layer2.parameters())
        
        # Apply changes to both layers
        layer1.apply_change()
        layer2.apply_change()
        
        # Verify parameters were updated
        final_params1 = sum(p.numel() for p in layer1.parameters())
        final_params2 = sum(p.numel() for p in layer2.parameters())
        
        # Clean up
        layer1.delete_update()
        layer2.delete_update()
        
    def test_complete_growth_cycle_with_multiple_containers(self):
        """Test a complete growth cycle from initialization to application across multiple container types."""
        # Initialize models
        residual_model = GrowingResidualMLP(
            in_features=self.input_shape,
            out_features=self.num_classes,
            num_features=32,
            hidden_features=16,
            num_blocks=2,
            activation=nn.ReLU()
        )
        
        mlp_model = GrowingMLP(
            in_features=self.input_shape[0] * self.input_shape[1] * self.input_shape[2],
            out_features=self.num_classes,
            hidden_size=32,
            number_hidden_layers=2,
            activation=nn.ReLU()
        )
        
        models = [residual_model, mlp_model]
        inputs = [self.x, self.x.view(self.batch_size, -1)]
        
        # Test complete growth cycle for each model
        for model, input_data in zip(models, inputs):
            # Store initial state
            initial_param_count = model.number_of_parameters()
            initial_output = model(input_data)
            
            # Initialize computation
            model.init_computation()
            
            # Run multiple batches to accumulate statistics
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for i in range(3):
                optimizer.zero_grad()
                output = model(input_data)
                loss = self.loss_fn(output, self.y_true)
                loss.backward()
                model.update_computation()
                
            # Compute optimal updates
            model.compute_optimal_updates()
            
            # Select and apply best update
            model.select_best_update()
            model.apply_change()
            
            # Verify changes were applied
            final_param_count = model.number_of_parameters()
            self.assertGreaterEqual(final_param_count, initial_param_count)
            
            # Verify model still works after growth
            final_output = model(input_data)
            self.assertEqual(final_output.shape, (self.batch_size, self.num_classes))
            
            # Clean up
            model.reset_computation()


if __name__ == "__main__":
    unittest.main()