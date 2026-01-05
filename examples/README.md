# GradMax Initialization Examples

This directory contains example scripts demonstrating the GradMax initialization method for progressive neural network growth.

## What is GradMax?

GradMax is an alternative initialization method for adding new neurons during progressive network growth. Compared to TINY (the default method):

- **TINY**: Uses projected gradients and optimizes both input (alpha) and output (omega) weights using SVD of S^(-1/2)N
- **GradMax**: Simpler approach that sets alpha=0 (zero input weights) and optimizes only output weights (omega) using SVD of M^T

GradMax focuses on maximizing the gradient norm of output weights, making it computationally simpler while still providing effective growth.

## Available Examples

### 1. `gradmax_mlp_tutorial.py`
A comprehensive tutorial demonstrating GradMax initialization on a simple MLP with synthetic data.

**Features:**
- Step-by-step progressive growth process
- Detailed documentation of each growth phase
- Visualization of training progress and parameter growth

**Usage:**
```bash
python examples/gradmax_mlp_tutorial.py
```

### 2. `compare_tiny_gradmax.py`
Side-by-side comparison of TINY and GradMax initialization methods.

**Features:**
- Trains two identical MLPs with different initialization methods
- Direct performance comparison
- Comparative visualization

**Usage:**
```bash
python examples/compare_tiny_gradmax.py
```

## How to Use GradMax

To use GradMax initialization in your own code, simply pass `initialization_method="gradmax"` when calling `compute_optimal_updates()`:

```python
from gromo.containers.growing_mlp import GrowingMLP

# Create model
model = GrowingMLP(
    in_features=10,
    out_features=3,
    hidden_size=2,
    number_hidden_layers=2,
    device=device,
)

# Training loop
model.train()
for data, target in train_loader:
    # ... forward/backward pass ...
    model.update_computation()

# Growth with GradMax
model.compute_optimal_updates(
    maximum_added_neurons=2,
    initialization_method="gradmax",  # Use GradMax instead of TINY
)
model.dummy_select_update()
model.set_scaling_factor(1.0)
model.apply_change()
model.reset_computation()
```

## Key Parameters

- `initialization_method`: Choose between `"tiny"` (default) or `"gradmax"`
- `maximum_added_neurons`: Maximum number of neurons to add per growth step
- `statistical_threshold`: Threshold for singular value selection (default: 1e-3)

## Results

Both methods produce effective growth, but with different characteristics:

- **TINY**: More sophisticated optimization, potentially better theoretical guarantees
- **GradMax**: Simpler, faster computation, good empirical performance

See the comparison script for detailed performance metrics.

## Dependencies

- PyTorch
- matplotlib (for visualization)
- GroMo library with GradMax support

## References

1. **TINY**: "Growing Tiny Networks" (arXiv:2405.19816)
2. **GradMax**: "GradMax: Growing Neural Networks using Gradient Information" (arXiv:2201.05125)

For more information about the GroMo library, see the [main documentation](https://theorudkiewicz.github.io/gromo/).

