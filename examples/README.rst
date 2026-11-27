--------
Examples
--------

Those examples demonstrate how to use growing modules in various contexts.

Recommended starting points:

- ``plot_growing_module_tutorial.py``: a minimal one-hidden-layer example.
- ``plot_growing_container_tutorial.py``: container-level growth with training,
  statistics, line search, and application of growth.
- ``plot_gradmax_tutorial.py``: a GradMax-oriented example that maps the
  paper's initialization style onto the current boolean-flag API.
- ``debug_conv_linear_flatten_transition.py``: a diagnostic example for the
  convolution-to-linear transition.

For reference, the paper-level configurations discussed in the codebase are:

- TINY: ``compute_delta=True``, ``use_covariance=True``, ``alpha_zero=False``,
  ``use_projection=True``
- GradMax: ``compute_delta=False``, ``use_covariance=False``, ``alpha_zero=True``,
  ``use_projection=False``
