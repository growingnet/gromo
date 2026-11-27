=====
Tools
=====

.. automodule:: gromo.utils.tools
   :no-index:

.. currentmodule:: gromo.utils.tools

----------------
Matrix inversion
----------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    sqrt_inverse_matrix_semi_positive

-------
Helpers
-------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    compute_optimal_added_parameters
    compute_output_shape_conv
    compute_mask_tensor_t
    spectrum_summary

----------
Thresholds
----------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    resolve_threshold
    resolve_threshold_rule

---------
Shrinkage
---------

.. automodule:: gromo.utils.shrinkage
   :no-index:

.. currentmodule:: gromo.utils.shrinkage

.. autosummary::
    :toctree: generated/
    :template: function.rst

    fixed_shrinkage
    weighted_ledoit_wolf
    reduce_fourth_moment

.. autosummary::
    :toctree: generated/
    :template: class.rst

    LedoitWolfEstimate
    ReduceMoments
