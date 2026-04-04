Variance-transfer initialization
=================================

This section describes the variance-transfer initialization scheme from
[Yuan2023]_, adapted to the GroMo codebase.  It covers two independent
mechanisms -- *neuron pairing* for function preservation and *rescaling
strategies* for variance stability -- and their combined analysis.

Setup: ResNet BasicBlock structure
-----------------------------------

A ``Conv2dGrowingBlock`` consists of two convolution layers with a residual
connection:

.. code-block:: text

   x ───────────────────────────────────────────────────── (+) ── out
    |                                                       ^
    └─> [PreAct] -> [Conv1: W1] -> [MidAct] -> [Conv2: W2] ─┘

with:

- :math:`W_1 \in \mathbb{R}^{h \times C_{\text{in}} \times k \times k}` (first
  conv, ``in_channels`` :math:`\to` ``hidden_channels``)
- :math:`W_2 \in \mathbb{R}^{C_{\text{out}} \times h \times k \times k}`
  (second conv, ``hidden_channels`` :math:`\to` ``out_channels``)
- :math:`h` is the current hidden channel count
- :math:`C_{\text{in}}, C_{\text{out}}` are fixed by the residual connection

Growth **adds neurons to the hidden dimension** :math:`h \to h'`:

- :math:`W_1` grows along its **output** dimension (more output channels)
- :math:`W_2` grows along its **input** dimension (more input channels)


Part 1: Neuron pairing -- :math:`(V,V)/(Z,-Z)`
-------------------------------------------------

This section describes **how new neurons are added**, independently of how
existing weights are rescaled (Part 2).

Structure
^^^^^^^^^^

We add :math:`\Delta h` hidden channel **pairs** so that
:math:`h_{t+1} = h_t + 2\Delta h`.  Each new neuron is duplicated into a pair
whose net contribution cancels at initialization.

New weight matrices:

.. math::

   V \in \mathbb{R}^{\Delta h \times C_{\text{in}} \times k \times k}
   \qquad\text{(output extension of Conv1)}

   Z \in \mathbb{R}^{C_{\text{out}} \times \Delta h \times k \times k}
   \qquad\text{(input extension of Conv2)}

Assembled layers after growth (where :math:`\alpha, \beta` are rescaling
factors from Part 2):

.. math::

   W_1^{(t+1)} =
   \begin{bmatrix}
   \alpha\, W_1^{(t)} \\
   V \\
   V
   \end{bmatrix}
   \in \mathbb{R}^{(h_t + 2\Delta h) \times C_{\text{in}} \times k \times k}

   W_2^{(t+1)} =
   \begin{bmatrix}
   \beta\, W_2^{(t)} & Z & -Z
   \end{bmatrix}
   \in \mathbb{R}^{C_{\text{out}} \times (h_t + 2\Delta h) \times k \times k}


Function preservation
^^^^^^^^^^^^^^^^^^^^^^

The new neurons produce activations :math:`\sigma(V * x)`, duplicated
identically.  The second layer reads them via :math:`(Z, -Z)`:

.. math::

   Z \cdot \sigma(V * x) + (-Z) \cdot \sigma(V * x) = 0

Therefore the block output is preserved up to rescaling:

.. math::

   \text{Block}_{t+1}(x) = \alpha\,\beta\;\text{Block}_t(x)

Exact function preservation requires :math:`\alpha\,\beta = 1`.


Part 2: Rescaling strategies
------------------------------

Three strategies are supported for choosing the rescaling factors
:math:`\alpha` (Conv1) and :math:`\beta` (Conv2).  They are **independent** of
the neuron-pairing mechanism.


Strategy A: ``"default_vt"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default strategy from [Yuan2023]_ (Section 3.1, Table 1).  Rescaling factors
depend only on the width ratio, not on actual weight statistics.

Conv1's input (:math:`C_{\text{in}}`) is not extended, so Conv1 is not
rescaled:

.. math::

   \alpha = 1, \qquad
   \beta = \sqrt{\frac{h_t}{h_{t+1}}}


Strategy B: ``"vt_constraint_old_shape"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the paper Appendix (Theorem 1).  Uses **actual weight statistics** to
enforce :math:`\operatorname{Var}[W] = 1/\text{fan_in_old}` after rescaling:

.. math::

   \alpha = \frac{1}{\sqrt{C_{\text{in}} \cdot k^2 \cdot
            \operatorname{Var}[W_1^{(t)}]}}, \qquad
   \beta  = \frac{1}{\sqrt{h_t \cdot k^2 \cdot
            \operatorname{Var}[W_2^{(t)}]}}


Strategy C: ``"vt_constraint_new_shape"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Like Strategy B but targets :math:`1/\text{fan_in_new}` instead of
:math:`1/\text{fan\_in\_old}`:

.. math::

   \alpha = \frac{1}{\sqrt{C_{\text{in}} \cdot k^2 \cdot
            \operatorname{Var}[W_1^{(t)}]}}, \qquad
   \beta  = \frac{1}{\sqrt{h_{t+1} \cdot k^2 \cdot
            \operatorname{Var}[W_2^{(t)}]}}

Note: :math:`\alpha` is the same as in Strategy B (Conv1's fan-in does not
change during growth).  Only :math:`\beta` differs: :math:`h_t` (B) vs
:math:`h_{t+1}` (C) in the denominator.


Summary table
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Strategy
     - :math:`\alpha` (Conv1)
     - :math:`\beta` (Conv2)
   * - A: Default VT
     - :math:`1`
     - :math:`\sqrt{h_t / h_{t+1}}`
   * - B: VT old shape
     - :math:`1 / \sqrt{C_{\text{in}} k^2\, \operatorname{Var}[W_1]}`
     - :math:`1 / \sqrt{h_t\, k^2\, \operatorname{Var}[W_2]}`
   * - C: VT new shape
     - :math:`1 / \sqrt{C_{\text{in}} k^2\, \operatorname{Var}[W_1]}`
     - :math:`1 / \sqrt{h_{t+1}\, k^2\, \operatorname{Var}[W_2]}`


BatchNorm adjustment
^^^^^^^^^^^^^^^^^^^^^

When a layer's weights are scaled by factor :math:`c`, the BatchNorm running
statistics must be adjusted accordingly:

.. math::

   \mu \leftarrow c\,\mu, \qquad
   \sigma^2 \leftarrow c^2\,\sigma^2


Part 3: Combined analysis
---------------------------

This section analyses the resulting weight and activation variances after
applying a rescaling strategy together with :math:`(V,V)/(Z,-Z)` neuron
pairing.

Definitions
^^^^^^^^^^^^

Consider the forward pass through one BasicBlock after a growth step.
Let :math:`x_{\text{pre}} = \sigma_{\text{pre}}(x)` be the pre-activated input
to Conv1.

**Hidden activations** :math:`u` (output of Conv1, input to Conv2):

- :math:`u'`: component from old weights:
  :math:`u' = \alpha\, W_1^{(t)} * x_{\text{pre}}`
- :math:`u''`: component from new weights:
  :math:`u'' = V * x_{\text{pre}}` (duplicated as :math:`(V, V)`)

After mid-activation:
:math:`\sigma(u) = [\sigma(u'),\; \sigma(u''),\; \sigma(u'')]`

**Block output** :math:`y` (output of Conv2, before residual addition):

- :math:`y' = \beta\, W_2^{(t)} \cdot \sigma(u')` (old pathway)
- :math:`y'' = Z \cdot \sigma(u'') + (-Z) \cdot \sigma(u'') = 0` (new pathway
  cancels at init)

At initialization :math:`y'' = 0`, so
:math:`y = y' = \alpha\,\beta\;\text{Block}_t(x)`.


Activation variance
^^^^^^^^^^^^^^^^^^^^

Assuming inputs have unit variance and weights are independent:

**Hidden activations:**

.. math::

   \operatorname{Var}[u'] &= \alpha^2\, C_{\text{in}}\, k^2\,
   \operatorname{Var}[W_1^{(t)}]

   \operatorname{Var}[u''] &= C_{\text{in}}\, k^2\, \operatorname{Var}[V] = 1
   \quad\text{(by Kaiming init of } V \text{)}

**Block output** (old pathway):

.. math::

   \operatorname{Var}[y'] = \beta^2\, h_t\, k^2\,
   \operatorname{Var}[W_2^{(t)}]\, \operatorname{Var}[\sigma(u')]

**New pathway at init:** :math:`\operatorname{Var}[y'']_{\text{init}} = 0` (by
:math:`(Z,-Z)` cancellation).

**New pathway after first gradient step** (symmetry broken):

.. math::

   \operatorname{Var}[y'']_{\text{1st}} \approx 2\Delta h \cdot k^2 \cdot
   \operatorname{Var}[Z] \cdot \operatorname{Var}[\sigma(u'')]

Since :math:`y = y' + y''` and the two contributions are independent:

.. math::

   \operatorname{Var}[y]_{\text{init}} &= \operatorname{Var}[y']

   \operatorname{Var}[y]_{\text{1st}} &= \operatorname{Var}[y'] +
   \operatorname{Var}[y'']_{\text{1st}}


Strategy A
^^^^^^^^^^^

With :math:`\alpha = 1` and :math:`\beta = \sqrt{h_t/h_{t+1}}`:

.. math::

   \operatorname{Var}[u'] = C_{\text{in}}\, k^2\,
   \operatorname{Var}[W_1^{(t)}]

This equals 1 only if :math:`\operatorname{Var}[W_1^{(t)}] =
1/(C_{\text{in}} k^2)` already holds.
Strategy A preserves variance across growth steps only if the weights already
have the correct variance.


Strategy B
^^^^^^^^^^^

By construction:

.. math::

   \alpha^2\, \operatorname{Var}[W_1^{(t)}] &= \frac{1}{C_{\text{in}} k^2}
   \implies \operatorname{Var}[u'] = 1

   \beta^2\, \operatorname{Var}[W_2^{(t)}] &= \frac{1}{h_t\, k^2}
   \implies \operatorname{Var}[y'] = 1

At init :math:`\operatorname{Var}[y] = 1` (exact).
After one gradient step:

.. math::

   \operatorname{Var}[y]_{\text{1st}} = 1 + \frac{2\Delta h}{h_{t+1}} > 1

The excess shrinks as :math:`\Delta h / h_{t+1} \to 0`.

**Merged weight variances:**

.. math::

   \operatorname{Var}[W_1^{(t+1)}] &= \frac{1}{C_{\text{in}} k^2}
   \quad\text{(exact)}

   \operatorname{Var}[W_2^{(t+1)}] &\approx \frac{1}{h_{t+1}\, k^2}
   \quad\text{(approximate, improves as } \Delta h/h_t \to 0 \text{)}


Strategy C
^^^^^^^^^^^

By construction:

.. math::

   \alpha^2\, \operatorname{Var}[W_1^{(t)}] &= \frac{1}{C_{\text{in}} k^2}
   \implies \operatorname{Var}[u'] = 1

   \beta^2\, \operatorname{Var}[W_2^{(t)}] &= \frac{1}{h_{t+1}\, k^2}
   \implies \operatorname{Var}[y'] = \frac{h_t}{h_{t+1}}

At init :math:`\operatorname{Var}[y] = h_t/h_{t+1} < 1`.
After one gradient step:

.. math::

   \operatorname{Var}[y]_{\text{1st}} = \frac{h_t}{h_{t+1}} +
   \frac{2\Delta h}{h_{t+1}} = \frac{h_t + 2\Delta h}{h_{t+1}} = 1
   \quad\text{(exact)}

**Merged weight variances:**

.. math::

   \operatorname{Var}[W_1^{(t+1)}] &= \frac{1}{C_{\text{in}} k^2}
   \quad\text{(exact)}

   \operatorname{Var}[W_2^{(t+1)}] &= \frac{1}{h_{t+1}\, k^2}
   \quad\text{(exact)}


Summary: resulting variances after growth
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Weight variances:**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Strategy
     - :math:`\operatorname{Var}[W_1^{(t+1)}]`
     - :math:`\operatorname{Var}[W_2^{(t+1)}]`
   * - A
     - depends on prior :math:`\operatorname{Var}[W_1^{(t)}]`
     - depends on prior :math:`\operatorname{Var}[W_2^{(t)}]`
   * - B
     - :math:`= 1/(C_{\text{in}} k^2)` (exact)
     - :math:`\approx 1/(h_{t+1} k^2)` (approximate)
   * - C
     - :math:`= 1/(C_{\text{in}} k^2)` (exact)
     - :math:`= 1/(h_{t+1} k^2)` (exact)

**Activation variances:**

.. list-table::
   :header-rows: 1
   :widths: 15 25 25 25

   * -
     - :math:`\operatorname{Var}[y]_{\text{init}}`
     - :math:`\operatorname{Var}[y]_{\text{1st step}}`
     - Trade-off
   * - A
     - depends on prior weights
     - depends on prior weights
     - no correction
   * - B
     - :math:`1` (exact)
     - :math:`1 + 2\Delta h / h_{t+1}` (> 1)
     - stable at init, slight excess after
   * - C
     - :math:`h_t / h_{t+1}` (< 1)
     - :math:`1` (exact)
     - small init deficit, exact after 1 step


Implementation
--------------

The variance-transfer features are exposed through
:meth:`~gromo.modules.growing_module.GrowingModule.create_layer_extensions` via
two parameters:

- ``rescaling``: one of ``None``, ``"default_vt"``,
  ``"vt_constraint_old_shape"``, ``"vt_constraint_new_shape"``
- ``neuron_pairing``: one of ``None``, ``"vv_z_negz"``

These can also be called independently as standalone methods for the FOGRO
growth path, where extensions are created by
``compute_optimal_added_parameters`` and trimmed by
``sub_select_optimal_added_parameters`` before rescaling and pairing are
applied:

- :meth:`~gromo.modules.growing_module.GrowingModule.apply_rescaling`
- :meth:`~gromo.modules.growing_module.GrowingModule.apply_neuron_pairing`

All parameters default to ``None``, preserving full backward compatibility.


References
-----------

.. [Yuan2023] Yuan et al., "Accelerated Training via Incrementally Growing
   Neural Networks using Variance Transfer and Learning Rate Adaptation",
   NeurIPS 2023.
