Extension initialization
========================

When a layer of neurons is grown, new incoming and outgoing connections are
created via
:meth:`~gromo.modules.growing_module.GrowingModule.create_layer_extensions`.
The method accepts an ``output_extension_init`` and an ``input_extension_init``
argument that control how the new weight tensors are filled.

The execution order inside ``create_layer_extensions`` is:

1. **Rescaling** -- existing weights are rescaled in-place (see
   :doc:`variance_transfer`).
2. **Extension creation** -- physical extension layers are allocated.
3. **Initialisation** -- extension weights are initialised (this section).
4. **Neuron pairing** -- extensions are doubled via :math:`(V,V)/(Z,-Z)` (see
   :doc:`variance_transfer`).

Available initialization methods
---------------------------------

Each initialization method is a callable with the signature
``(tensor, reference_tensor, fan_in)`` where *tensor* is the weight (or bias)
to initialize, *reference_tensor* is the existing weight (or bias) of the base
layer, and *fan_in* is the fan-in of the layer after including the extension.

``"kaiming"``
^^^^^^^^^^^^^

Standard Kaiming uniform initialization.  Samples from

.. math::

   \mathcal{U}\!\left(-\sqrt{\frac{6}{\text{fan_in}}},\;
   \sqrt{\frac{6}{\text{fan_in}}}\right)

where :math:`\text{fan_in}` is the total fan-in of the layer (base + extension).
This is the PyTorch default (``torch.nn.init.kaiming_uniform_`` with
:math:`a=0`) and ensures :math:`\operatorname{Var}[W] = 2/\text{fan_in}`.

``"copy_uniform"``
^^^^^^^^^^^^^^^^^^

Samples from a uniform distribution whose bounds are derived from the
*empirical* standard deviation of the existing weights:

.. math::

   \mathcal{U}\!\left(-\sqrt{3}\;\sigma_{W},\;
   \sqrt{3}\;\sigma_{W}\right)

where :math:`\sigma_{W} = \operatorname{std}(W_{\text{ref}})` is computed over
all elements of ``reference_tensor``.
The resulting distribution has variance :math:`\sigma_{W}^{2}`, matching the
existing weight distribution.

This is the default initialization in ``create_layer_extensions``.  It is
particularly useful when the existing weights have drifted from the Kaiming
scale during training: ``copy_uniform`` initializes the extension in the same
statistical regime as the existing weights, avoiding a sudden variance
mismatch in the merged layer.

**Fallback:** when the reference tensor is ``None``, has fewer than 2 elements,
or has zero variance (e.g. all weights identical), ``copy_uniform`` falls back
to ``kaiming`` initialization.

``"zeros"``
^^^^^^^^^^^

Fills the extension with zeros (``torch.nn.init.zeros_``).  Useful for
debugging or when the extension should have no initial contribution.

Comparison
----------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Variance of new weights
     - When to use
   * - ``kaiming``
     - :math:`2/\text{fan_in}` (theoretical Kaiming)
     - Fresh or well-scaled weights; standard baseline
   * - ``copy_uniform``
     - :math:`\operatorname{Var}[W_{\text{existing}}]` (empirical)
     - After training when weights have drifted from Kaiming scale
   * - ``zeros``
     - 0
     - Debugging, or combined with neuron pairing

Interaction with variance-transfer rescaling
---------------------------------------------

When a rescaling strategy is active (see :doc:`variance_transfer`), the
existing weights are rescaled *before* the extension is initialized.  This
ordering matters for ``copy_uniform``: the reference variance
:math:`\sigma_{W}^{2}` is read from the already-rescaled weights, so the new
extension matches the post-rescaling regime.

For ``kaiming`` initialization this ordering has no effect, since the bounds
depend only on ``fan_in``.

Adding custom initialization methods
--------------------------------------

New methods can be registered by adding an entry to the ``known_inits``
dictionary inside ``create_layer_extensions``.  The callable must accept three
positional arguments:

.. code-block:: python

   def my_init(
       tensor: torch.Tensor,
       reference_tensor: torch.Tensor | None,
       fan_in: int,
   ) -> None:
       ...

The callable may also modify ``reference_tensor`` in-place if needed (e.g. to
perturb existing weights at the same time).
