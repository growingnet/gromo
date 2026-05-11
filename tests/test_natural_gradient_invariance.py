r"""Natural-gradient invariance tests for the empirical-Fisher integration.

Setup
-----
We work with a 4-layer purely linear network (no biases, identity post-
activations):

    x ---W_1---> h_1 ---W_2---> h_2 ---W_3---> h_3 ---W_4---> y

The network admits a well-known scaling symmetry: if we multiply ``W_1`` by a
positive scalar ``c`` and divide ``W_4`` by ``c`` (so that the leading factor
introduced at the input is undone at the output), the network function ``y(x)``
is exactly preserved. Note that the natural symmetry is ``W_4 <- W_4 / c``
rather than ``W_4 <- c * W_4``: the latter would change the output by a
factor of ``c**2``.

Under that rescaling, we focus on layer L3 (the third hidden layer, with
previous_module = L2). Direct computation gives the per-sample scalings:

    h_2 (= L3.input)            ~ c       (because L1's output is scaled by c
                                          and W_2 is unchanged, so h_2 ~ c)
    pre_3                       ~ c
    dA_3 = d(loss)/d(pre_3)     ~ 1/c    (chain rule: scaling pre_3 by c
                                          divides its gradient by c)

so the recorded statistics scale as

    tensor_s_growth (= L2.tensor_s)              ~ c**2
    tensor_m_prev   (= L3.tensor_m_prev)         ~ 1     (= c * 1/c)
    tensor_s        (= L3.tensor_s)              ~ c**2
    tensor_m        (= L3.tensor_m)              ~ 1
    covariance_loss_gradient (L3.cov_grad)       ~ 1/c**2.

From these, the predictions verified below are:

* ``compute_optimal_delta``::

      use_fisher=False:  delta_raw = (S^{-1} M)^T               ~ 1/c**2
      use_fisher=True:   delta_raw = E^{-1} (S^{-1} M)^T         ~ 1   (invariant)

* ``compute_optimal_added_parameters`` singular values
  (``eigenvalues_extension``, with ``compute_delta=False`` and
  ``use_projection=False`` so the SVD operates on ``-tensor_m_prev``)::

      use_covariance=False, use_fisher=False:
          P = N                                 ~ 1               (invariant)
      use_covariance=True,  use_fisher=False:
          P = S^{-1/2} N                        ~ 1/c             (scales by 1/c)
      use_covariance=True,  use_fisher=True:
          P = S^{-1/2} N E^{+/2}                ~ 1               (invariant)
"""

import torch

from gromo.modules.conv2d_growing_module import RestrictedConv2dGrowingModule
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase


class TestNaturalGradientInvariance(TorchTestCase):
    SEED = 42
    BATCH_SIZE = 16
    # in -> L1_out -> L2_out -> L3_out -> L4_out
    DIMS = (3, 4, 5, 4, 2)
    # Non-trivial scaling factor; kept moderate to avoid driving any
    # eigenvalue below the numerical / statistical thresholds used inside
    # ``compute_optimal_added_parameters``.
    SCALE = 2.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_net(self, c: float):
        """Build the 4-layer linear chain with the rescaling W_1 *= c,
        W_4 /= c applied to the (otherwise deterministic) weights.

        L1 and L4 are plain `nn.Linear` and are frozen -- we only collect
        statistics on L3 (with previous_module = L2). The same input batch
        and target are reused across calls.
        """
        device = global_device()
        d0, d1, d2, d3, d4 = self.DIMS

        torch.manual_seed(self.SEED)
        l1 = torch.nn.Linear(d0, d1, bias=False, device=device)
        l2 = LinearGrowingModule(d1, d2, use_bias=False, device=device, name="l2")
        l3 = LinearGrowingModule(
            d2, d3, use_bias=False, previous_module=l2, device=device, name="l3"
        )
        l4 = torch.nn.Linear(d3, d4, bias=False, device=device)

        with torch.no_grad():
            l1.weight.mul_(c)
            l4.weight.mul_(1.0 / c)

        for p in l1.parameters():
            p.requires_grad_(False)
        for p in l4.parameters():
            p.requires_grad_(False)

        torch.manual_seed(self.SEED + 1)
        x = torch.randn(self.BATCH_SIZE, d0, device=device)
        target = torch.randn(self.BATCH_SIZE, d4, device=device)

        return l1, l2, l3, l4, x, target

    def _collect_stats(self, l1, l2, l3, l4, x, target):
        """Run a single forward+backward and update L3's statistics."""
        l3.init_computation()
        net = torch.nn.Sequential(l1, l2, l3, l4)
        y = net(x)
        loss = 0.5 * ((y - target) ** 2).sum()
        loss.backward()
        l3.update_computation()

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------

    def test_network_output_is_c_invariant(self):
        """Sanity: the rescaling W_1 *= c, W_4 /= c preserves ``y(x)``.

        This is a precondition for the rest of the file: if the rescaled
        network produced a different output, the gradients would also differ
        and the predicted scalings of the statistics would not hold.
        """
        l1_a, l2_a, l3_a, l4_a, x_a, _ = self._build_net(c=1.0)
        y_a = torch.nn.Sequential(l1_a, l2_a, l3_a, l4_a)(x_a).detach()

        l1_b, l2_b, l3_b, l4_b, x_b, _ = self._build_net(c=self.SCALE)
        y_b = torch.nn.Sequential(l1_b, l2_b, l3_b, l4_b)(x_b).detach()

        # Same input must give the same output.
        assert torch.allclose(x_a, x_b)
        self.assertAllClose(y_a, y_b, atol=1e-5, rtol=1e-5)

    # ------------------------------------------------------------------
    # Predicted statistics scalings (also a useful debugging aid)
    # ------------------------------------------------------------------

    def test_recorded_statistics_scale_as_predicted(self):
        """Verify the per-statistic scalings used in the analytical predictions
        of the two main tests below.
        """
        c = self.SCALE

        l1, l2, l3, l4, x, target = self._build_net(c=1.0)
        self._collect_stats(l1, l2, l3, l4, x, target)
        s_growth_ref = l3.tensor_s_growth().detach().clone()
        m_prev_ref = l3.tensor_m_prev().detach().clone()
        s_ref = l3.tensor_s().detach().clone()
        m_ref = l3.tensor_m().detach().clone()
        cov_ref = l3.covariance_loss_gradient().detach().clone()

        l1, l2, l3, l4, x, target = self._build_net(c=c)
        self._collect_stats(l1, l2, l3, l4, x, target)
        # tensor_s_growth ~ c**2
        self.assertAllClose(l3.tensor_s_growth(), (c * c) * s_growth_ref, atol=1e-5)
        # tensor_m_prev ~ 1 (invariant)
        self.assertAllClose(l3.tensor_m_prev(), m_prev_ref, atol=1e-5)
        # tensor_s ~ c**2
        self.assertAllClose(l3.tensor_s(), (c * c) * s_ref, atol=1e-5)
        # tensor_m ~ 1
        self.assertAllClose(l3.tensor_m(), m_ref, atol=1e-5)
        # covariance_loss_gradient ~ 1/c**2
        self.assertAllClose(
            l3.covariance_loss_gradient(),
            cov_ref / (c * c),
            atol=1e-5,
            rtol=1e-4,
        )

    # ------------------------------------------------------------------
    # Test 1: optimal_delta natural-gradient invariance
    # ------------------------------------------------------------------

    def test_optimal_delta_natural_gradient_invariance(self):
        r"""Without Fisher, ``delta_raw`` scales by ``1/c**2``; with Fisher
        it is invariant under the natural-gradient reparameterization
        ``W_1 -> c * W_1, W_4 -> W_4 / c``.
        """
        c = self.SCALE

        # --- reference run (c = 1) ---
        l1, l2, l3, l4, x, target = self._build_net(c=1.0)
        self._collect_stats(l1, l2, l3, l4, x, target)
        delta_no_f_ref, _, _ = l3.compute_optimal_delta(use_fisher=False, update=False)
        delta_f_ref, _, _ = l3.compute_optimal_delta(use_fisher=True, update=False)
        delta_no_f_ref = delta_no_f_ref.detach().clone()
        delta_f_ref = delta_f_ref.detach().clone()
        # Sanity: the reference solutions should be non-trivially non-zero
        # (otherwise the rescaling test below is vacuous).
        assert delta_no_f_ref.norm().item() > 1e-3
        assert delta_f_ref.norm().item() > 1e-3

        # --- rescaled run (c = SCALE) ---
        l1, l2, l3, l4, x, target = self._build_net(c=c)
        self._collect_stats(l1, l2, l3, l4, x, target)
        delta_no_f_resc, _, _ = l3.compute_optimal_delta(use_fisher=False, update=False)
        delta_f_resc, _, _ = l3.compute_optimal_delta(use_fisher=True, update=False)

        # Without Fisher: delta_resc ~ delta_ref / c**2.
        self.assertAllClose(
            delta_no_f_resc,
            delta_no_f_ref / (c * c),
            atol=1e-5,
            rtol=1e-4,
            msg="Without Fisher, delta_raw should scale by 1/c**2 under the "
            "natural-gradient reparameterization.",
        )
        # And the rescaled delta must clearly differ from the reference (i.e.
        # the rescaling actually had an effect).
        self.assertGreater(
            (delta_no_f_resc - delta_no_f_ref).abs().max().item(),
            1e-3,
            "Without Fisher, delta_raw should change under the rescaling; "
            "if it does not, either the rescaling is not effective or the "
            "reference delta is too small to make the test meaningful.",
        )

        # With Fisher: delta_raw is invariant.
        self.assertAllClose(
            delta_f_resc,
            delta_f_ref,
            atol=1e-5,
            rtol=1e-4,
            msg="With Fisher (use_fisher=True), delta_raw must be invariant "
            "under the natural-gradient reparameterization.",
        )

    # ------------------------------------------------------------------
    # Test 2: eigenvalues_extension natural-gradient invariance
    # ------------------------------------------------------------------

    def test_eigenvalues_extension_natural_gradient_invariance(self):
        r"""For ``compute_optimal_updates(compute_delta=False)`` on L3, the
        singular values returned in ``eigenvalues_extension`` scale as:

        * use_covariance=False, use_fisher=False  ->  invariant
        * use_covariance=True,  use_fisher=False  ->  scales by 1/c
        * use_covariance=True,  use_fisher=True   ->  invariant
        """
        c = self.SCALE

        def run(c_val: float, *, use_covariance: bool, use_fisher: bool):
            l1, l2, l3, l4, x, target = self._build_net(c=c_val)
            self._collect_stats(l1, l2, l3, l4, x, target)
            l3.compute_optimal_updates(
                compute_delta=False,
                use_covariance=use_covariance,
                use_fisher=use_fisher,
                use_projection=False,
            )
            return l3.eigenvalues_extension.detach().clone()

        # Singular values come back sorted in non-increasing order from
        # `torch.linalg.svd`; we sort defensively in case of ties.
        def sort_desc(t: torch.Tensor) -> torch.Tensor:
            return t.sort(descending=True).values

        # --- 1) no covariance, no Fisher: P = N ~ 1, sigmas invariant. ---
        eig_ref_1 = run(1.0, use_covariance=False, use_fisher=False)
        eig_resc_1 = run(c, use_covariance=False, use_fisher=False)
        self.assertEqual(eig_ref_1.shape, eig_resc_1.shape)
        # Sanity: ref must be non-trivial.
        assert eig_ref_1.max().item() > 1e-3
        self.assertAllClose(
            sort_desc(eig_resc_1),
            sort_desc(eig_ref_1),
            atol=1e-5,
            rtol=1e-4,
            msg="use_covariance=False, use_fisher=False: eigenvalues should "
            "be c-invariant.",
        )

        # --- 2) covariance only: P = S^{-1/2} N ~ 1/c, sigmas scale by 1/c. ---
        eig_ref_2 = run(1.0, use_covariance=True, use_fisher=False)
        eig_resc_2 = run(c, use_covariance=True, use_fisher=False)
        self.assertEqual(eig_ref_2.shape, eig_resc_2.shape)
        assert eig_ref_2.max().item() > 1e-3
        self.assertAllClose(
            sort_desc(eig_resc_2),
            sort_desc(eig_ref_2) / c,
            atol=1e-5,
            rtol=1e-4,
            msg="use_covariance=True, use_fisher=False: eigenvalues should scale by 1/c.",
        )
        self.assertGreater(
            (sort_desc(eig_resc_2) - sort_desc(eig_ref_2)).abs().max().item(),
            1e-3,
            "use_covariance=True, use_fisher=False: eigenvalues must change "
            "under the rescaling.",
        )

        # --- 3) covariance + Fisher: P = S^{-1/2} N E^{+/2} ~ 1, invariant. ---
        eig_ref_3 = run(1.0, use_covariance=True, use_fisher=True)
        eig_resc_3 = run(c, use_covariance=True, use_fisher=True)
        self.assertEqual(eig_ref_3.shape, eig_resc_3.shape)
        assert eig_ref_3.max().item() > 1e-3
        self.assertAllClose(
            sort_desc(eig_resc_3),
            sort_desc(eig_ref_3),
            atol=1e-5,
            rtol=1e-4,
            msg="use_covariance=True, use_fisher=True: eigenvalues should "
            "be c-invariant.",
        )


class TestNaturalGradientInvarianceRestrictedConv2d(TorchTestCase):
    r"""Same scale-invariance check as ``TestNaturalGradientInvariance`` but for
    a chain of ``RestrictedConv2dGrowingModule``\ s.

    Layout: ``conv2d_a -> RestrictedConv2dGrowingModule_b
                       -> RestrictedConv2dGrowingModule_c
                       -> conv2d_d``.

    All convolutions are 3x3 with padding=1 and stride=1 so that the spatial
    dimension is preserved end-to-end. Like in the linear case, the network is
    purely linear (identity post-activations everywhere, no biases) so the
    rescaling ``W_a *= c, W_d /= c`` exactly preserves the network output.

    Conv2d statistics scale identically to the linear case because all
    convolutions are linear in the input, so the unfolded inputs pick up the
    same factor as the input activations:

        L_b.input  ~ c       L_c.input  ~ c       pre_c ~ c       dA_c ~ 1/c
        tensor_s_growth (= L_b.tensor_s)         ~ c**2
        tensor_m_prev   (= L_c.tensor_m_prev)    ~ 1
        tensor_s        (= L_c.tensor_s)         ~ c**2
        tensor_m        (= L_c.tensor_m)         ~ 1
        covariance_loss_gradient (L_c)           ~ 1/c**2.

    The predicted scalings of ``compute_optimal_delta`` and
    ``eigenvalues_extension`` are therefore identical to the linear case.
    """

    SEED = 42
    BATCH_SIZE = 4
    SPATIAL = (5, 5)
    # in -> L_a_out -> L_b_out -> L_c_out -> L_d_out
    CHANNELS = (2, 4, 5, 4, 2)
    SCALE = 2.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_net(self, c: float):
        r"""Build the 4-conv chain with the rescaling W_a *= c, W_d /= c.

        L_a and L_d are plain ``torch.nn.Conv2d`` (frozen). L_b and L_c are
        ``RestrictedConv2dGrowingModule``\ s, with ``L_c.previous_module = L_b``
        so that we collect statistics on L_c (with L_b providing
        ``tensor_s_growth``).
        """
        device = global_device()
        c0, c1, c2, c3, c4 = self.CHANNELS

        torch.manual_seed(self.SEED)
        l_a = torch.nn.Conv2d(
            c0, c1, kernel_size=3, padding=1, stride=1, bias=False, device=device
        )
        l_b = RestrictedConv2dGrowingModule(
            in_channels=c1,
            out_channels=c2,
            kernel_size=3,
            padding=1,
            stride=1,
            use_bias=False,
            device=device,
            name="l_b",
        )
        l_c = RestrictedConv2dGrowingModule(
            in_channels=c2,
            out_channels=c3,
            kernel_size=3,
            padding=1,
            stride=1,
            use_bias=False,
            previous_module=l_b,
            device=device,
            name="l_c",
        )
        l_d = torch.nn.Conv2d(
            c3, c4, kernel_size=3, padding=1, stride=1, bias=False, device=device
        )

        with torch.no_grad():
            l_a.weight.mul_(c)
            l_d.weight.mul_(1.0 / c)

        for p in l_a.parameters():
            p.requires_grad_(False)
        for p in l_d.parameters():
            p.requires_grad_(False)

        torch.manual_seed(self.SEED + 1)
        h, w = self.SPATIAL
        x = torch.randn(self.BATCH_SIZE, c0, h, w, device=device)
        target = torch.randn(self.BATCH_SIZE, c4, h, w, device=device)

        return l_a, l_b, l_c, l_d, x, target

    def _collect_stats(self, l_a, l_b, l_c, l_d, x, target):
        """Single forward+backward pass that populates L_c's statistics."""
        l_c.init_computation()
        net = torch.nn.Sequential(l_a, l_b, l_c, l_d)
        y = net(x)
        loss = 0.5 * ((y - target) ** 2).sum()
        loss.backward()
        l_c.update_computation()

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------

    def test_network_output_is_c_invariant(self):
        """Sanity: the rescaling W_a *= c, W_d /= c preserves ``y(x)``."""
        l_a_1, l_b_1, l_c_1, l_d_1, x_1, _ = self._build_net(c=1.0)
        y_1 = torch.nn.Sequential(l_a_1, l_b_1, l_c_1, l_d_1)(x_1).detach()

        l_a_2, l_b_2, l_c_2, l_d_2, x_2, _ = self._build_net(c=self.SCALE)
        y_2 = torch.nn.Sequential(l_a_2, l_b_2, l_c_2, l_d_2)(x_2).detach()

        assert torch.allclose(x_1, x_2)
        self.assertAllClose(y_1, y_2, atol=1e-5, rtol=1e-5)

    # ------------------------------------------------------------------
    # Predicted statistics scalings
    # ------------------------------------------------------------------

    def test_recorded_statistics_scale_as_predicted(self):
        c = self.SCALE

        l_a, l_b, l_c, l_d, x, target = self._build_net(c=1.0)
        self._collect_stats(l_a, l_b, l_c, l_d, x, target)
        s_growth_ref = l_c.tensor_s_growth().detach().clone()
        m_prev_ref = l_c.tensor_m_prev().detach().clone()
        s_ref = l_c.tensor_s().detach().clone()
        m_ref = l_c.tensor_m().detach().clone()
        cov_ref = l_c.covariance_loss_gradient().detach().clone()

        l_a, l_b, l_c, l_d, x, target = self._build_net(c=c)
        self._collect_stats(l_a, l_b, l_c, l_d, x, target)
        self.assertAllClose(
            l_c.tensor_s_growth(), (c * c) * s_growth_ref, atol=1e-5, rtol=1e-4
        )
        self.assertAllClose(l_c.tensor_m_prev(), m_prev_ref, atol=1e-5, rtol=1e-4)
        self.assertAllClose(l_c.tensor_s(), (c * c) * s_ref, atol=1e-5, rtol=1e-4)
        self.assertAllClose(l_c.tensor_m(), m_ref, atol=1e-5, rtol=1e-4)
        self.assertAllClose(
            l_c.covariance_loss_gradient(),
            cov_ref / (c * c),
            atol=1e-5,
            rtol=1e-4,
        )

    # ------------------------------------------------------------------
    # Test 1: optimal_delta natural-gradient invariance
    # ------------------------------------------------------------------

    def test_optimal_delta_natural_gradient_invariance(self):
        r"""Without Fisher, ``delta_raw`` scales by ``1/c**2``; with Fisher
        it is invariant under the natural-gradient reparameterization
        ``W_a *= c, W_d /= c``.
        """
        c = self.SCALE

        l_a, l_b, l_c, l_d, x, target = self._build_net(c=1.0)
        self._collect_stats(l_a, l_b, l_c, l_d, x, target)
        delta_no_f_ref, _, _ = l_c.compute_optimal_delta(use_fisher=False, update=False)
        delta_f_ref, _, _ = l_c.compute_optimal_delta(use_fisher=True, update=False)
        delta_no_f_ref = delta_no_f_ref.detach().clone()
        delta_f_ref = delta_f_ref.detach().clone()
        assert delta_no_f_ref.norm().item() > 1e-3
        assert delta_f_ref.norm().item() > 1e-3

        l_a, l_b, l_c, l_d, x, target = self._build_net(c=c)
        self._collect_stats(l_a, l_b, l_c, l_d, x, target)
        delta_no_f_resc, _, _ = l_c.compute_optimal_delta(use_fisher=False, update=False)
        delta_f_resc, _, _ = l_c.compute_optimal_delta(use_fisher=True, update=False)

        # Without Fisher: delta_raw scales by 1/c**2.
        self.assertAllClose(
            delta_no_f_resc,
            delta_no_f_ref / (c * c),
            atol=1e-5,
            rtol=1e-4,
            msg="Conv2d / no-Fisher: delta_raw should scale by 1/c**2.",
        )
        self.assertGreater(
            (delta_no_f_resc - delta_no_f_ref).abs().max().item(),
            1e-3,
            "Conv2d / no-Fisher: delta_raw must change under the rescaling.",
        )

        # With Fisher: delta_raw is invariant.
        self.assertAllClose(
            delta_f_resc,
            delta_f_ref,
            atol=1e-5,
            rtol=1e-4,
            msg="Conv2d / Fisher: delta_raw must be invariant.",
        )

    # ------------------------------------------------------------------
    # Test 2: eigenvalues_extension natural-gradient invariance
    # ------------------------------------------------------------------

    def test_eigenvalues_extension_natural_gradient_invariance(self):
        r"""For ``compute_optimal_updates(compute_delta=False)`` on L_c, the
        singular values returned in ``eigenvalues_extension`` scale as:

        * use_covariance=False, use_fisher=False  ->  invariant
        * use_covariance=True,  use_fisher=False  ->  scales by 1/c
        * use_covariance=True,  use_fisher=True   ->  invariant
        """
        c = self.SCALE

        def run(c_val: float, *, use_covariance: bool, use_fisher: bool):
            l_a, l_b, l_c, l_d, x, target = self._build_net(c=c_val)
            self._collect_stats(l_a, l_b, l_c, l_d, x, target)
            l_c.compute_optimal_updates(
                compute_delta=False,
                use_covariance=use_covariance,
                use_fisher=use_fisher,
                use_projection=False,
            )
            return l_c.eigenvalues_extension.detach().clone()

        def sort_desc(t: torch.Tensor) -> torch.Tensor:
            return t.sort(descending=True).values

        # 1) no covariance, no Fisher: P = N ~ 1, sigmas invariant.
        eig_ref_1 = run(1.0, use_covariance=False, use_fisher=False)
        eig_resc_1 = run(c, use_covariance=False, use_fisher=False)
        self.assertEqual(eig_ref_1.shape, eig_resc_1.shape)
        assert eig_ref_1.max().item() > 1e-3
        self.assertAllClose(
            sort_desc(eig_resc_1),
            sort_desc(eig_ref_1),
            atol=1e-5,
            rtol=1e-4,
            msg="Conv2d / use_covariance=False, use_fisher=False: eigenvalues "
            "should be c-invariant.",
        )

        # 2) covariance only: P = S^{-1/2} N ~ 1/c, sigmas scale by 1/c.
        eig_ref_2 = run(1.0, use_covariance=True, use_fisher=False)
        eig_resc_2 = run(c, use_covariance=True, use_fisher=False)
        self.assertEqual(eig_ref_2.shape, eig_resc_2.shape)
        assert eig_ref_2.max().item() > 1e-3
        self.assertAllClose(
            sort_desc(eig_resc_2),
            sort_desc(eig_ref_2) / c,
            atol=1e-5,
            rtol=1e-4,
            msg="Conv2d / use_covariance=True, use_fisher=False: eigenvalues "
            "should scale by 1/c.",
        )
        self.assertGreater(
            (sort_desc(eig_resc_2) - sort_desc(eig_ref_2)).abs().max().item(),
            1e-3,
            "Conv2d / use_covariance=True, use_fisher=False: eigenvalues must "
            "change under the rescaling.",
        )

        # 3) covariance + Fisher: P = S^{-1/2} N E^{+/2} ~ 1, invariant.
        eig_ref_3 = run(1.0, use_covariance=True, use_fisher=True)
        eig_resc_3 = run(c, use_covariance=True, use_fisher=True)
        self.assertEqual(eig_ref_3.shape, eig_resc_3.shape)
        assert eig_ref_3.max().item() > 1e-3
        self.assertAllClose(
            sort_desc(eig_resc_3),
            sort_desc(eig_ref_3),
            atol=1e-5,
            rtol=1e-4,
            msg="Conv2d / use_covariance=True, use_fisher=True: eigenvalues "
            "should be c-invariant.",
        )
