#!/usr/bin/env python3
"""Tests for the tagged BselCostanzi26, BselSunayama23, and BselModels APIs."""
from __future__ import annotations

import sys
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))

from systematics.BselCostanzi26 import BselCostanzi26  # noqa: E402
from systematics.BselSunayama23 import BselSunayama23  # noqa: E402
from systematics.bsel_profile import BselModels  # noqa: E402


def _costanzi_decimal_reference(R):
    """80-digit direct derivative of the Appendix C expression, at lob=40, z=0.3."""
    d = Decimal
    with localcontext() as context:
        context.prec = 80
        A, alpha, beta, gamma = map(d, ("0.1", "0.92", "-0.53", "4.1"))
        radius = d(str(R))
        r0 = ((d("40") / d("100")).ln() * d("0.2")).exp() * d("1.3")
        x = radius / r0
        x_gamma = (gamma * x.ln()).exp()
        smooth = (d(1) + x_gamma)
        value = d(1) + A * (alpha * x.ln()).exp() * (
            ((beta - alpha) / gamma) * smooth.ln()).exp()
        derivative = (A / r0) * ((alpha - 1) * x.ln()).exp() * (
            ((beta - alpha) / gamma - 1) * smooth.ln()).exp() * (
                alpha + beta * x_gamma)
        return float(value), float(derivative)


class TestBselModels(unittest.TestCase):
    def test_costanzi_value_and_derivative(self):
        model = BselCostanzi26(A=0.10, alpha=0.92, beta=-0.53, gamma=4.1)
        R = 1.0
        lob, z = 40.0, 0.3
        eps = 1.0e-6
        numerical = (model(R + eps, lob, z) - model(R - eps, lob, z)) / (2.0 * eps)
        np.testing.assert_allclose(model.derivative(R, lob, z), numerical, rtol=1.0e-8)
        self.assertEqual(float(model(0.0, lob, z)), 1.0)

    def test_costanzi_wall_validation(self):
        model = BselCostanzi26(A=0.1, alpha=0.92, beta=-0.53, gamma=4.1)
        lob, z = model.validate_wall([25.0, 40.0], [0.2, 0.4])
        self.assertEqual(lob.shape, (2,))
        self.assertEqual(z.shape, (2,))
        with self.assertRaises(ValueError):
            model.validate_wall([0.0], [0.2])

    def test_costanzi_decimal_reference_and_derivative_precision(self):
        model = BselCostanzi26(A=0.10, alpha=0.92, beta=-0.53, gamma=4.1)
        # These radii straddle R0=1.082319...; the 1e-8 point also measures
        # cancellation from forming B(R)-1 in the derivative near R=0.
        radii = np.array([1e-8, 1e-4, 0.5, 1.0, 3.0, 10.0])
        reference = np.array([_costanzi_decimal_reference(R) for R in radii])
        values = model(radii, 40.0, 0.3)
        derivatives = model.derivative(radii, 40.0, 0.3)
        value_error = np.max(np.abs((values - reference[:, 0]) / reference[:, 0]))
        derivative_error = np.max(np.abs(
            (derivatives - reference[:, 1]) / reference[:, 1]))
        self.assertLess(value_error, 3e-16)
        self.assertLess(derivative_error, 3e-8)
        # The physically sampled range is considerably more precise.
        np.testing.assert_allclose(derivatives[2:], reference[2:, 1],
                                   rtol=2e-14, atol=0.0)

    def test_costanzi_invalid_parameters_and_boundary(self):
        for params in ((np.nan, 0.92, -0.53, 4.1),
                       (0.1, 0.92, -0.53, 0.0),
                       (0.1, 0.92, -0.53, np.inf)):
            with self.assertRaises(ValueError):
                BselCostanzi26(*params)
        model = BselCostanzi26(0.1, 0.92, -0.53, 4.1)
        self.assertEqual(float(model(0.0, 40.0, 0.3)), 1.0)
        with self.assertRaises(ValueError):
            model(-1.0, 40.0, 0.3)
        with self.assertRaises(ValueError):
            model.derivative(0.0, 40.0, 0.3)
        for lob, z in ((0.0, 0.3), (40.0, -1.0)):
            with self.assertRaises(ValueError):
                model(1.0, lob, z)

    def test_costanzi_rejects_nonfinite_coordinates(self):
        model = BselCostanzi26(0.1, 0.92, -0.53, 4.1)
        with self.assertRaises(ValueError):
            model(np.nan, 40.0, 0.3)

    def test_costanzi_rejects_nonfinite_wall_even_at_zero_radius(self):
        model = BselCostanzi26(0.1, 0.92, -0.53, 4.1)
        with self.assertRaises(ValueError):
            model(1.0, np.nan, 0.3)
        with self.assertRaises(ValueError):
            model(0.0, 0.0, 0.3)

    def test_sunayama_branches(self):
        model = BselSunayama23(pi0=[1.2, 1.4], r0=[1.0, 2.0], c=0.1,
                                lambda_bin=[2, 5])
        np.testing.assert_allclose(model(0.5, 2), 1.6)
        np.testing.assert_allclose(model(1.0, 2), 2.2)
        np.testing.assert_allclose(model(2.0, 5), 2.4)
        np.testing.assert_allclose(model(4.0, 5), 2.4 + 0.1 * np.log(2.0))
        np.testing.assert_allclose(model.derivative(0.5, 2), 1.2)
        np.testing.assert_allclose(model.derivative(4.0, 5), 0.025)

    def test_sunayama_one_plus_pi_and_breakpoint(self):
        model = BselSunayama23(pi0=[1.2, 1.4], r0=[1.0, 2.0], c=0.1,
                               lambda_bin=[2, 5])
        radii = np.array([0.0, 0.5, 1.0, 1.0 + 1e-12, 4.0])
        values = model(radii, 2)
        reference = np.array([1.0, 1.6, 2.2,
                              2.2 + 0.1 * np.log1p(1e-12),
                              2.2 + 0.1 * np.log(4.0)])
        np.testing.assert_allclose(values, reference, rtol=2e-15, atol=0.0)
        np.testing.assert_allclose(model.derivative([0.5, 1.0, 1.0 + 1e-12], 2),
                                   [1.2, 1.2, 0.1 / (1.0 + 1e-12)],
                                   rtol=2e-15)
        np.testing.assert_allclose(model([0.5, 4.0], [2, 5]),
                                   [1.6, 2.4 + 0.1 * np.log(2.0)],
                                   rtol=2e-15)

    def test_sunayama_invalid_parameters_and_wall(self):
        for args in (([], [], 0.1), ([1.2], [0.0], 0.1),
                     ([1.2], [1.0], np.nan)):
            with self.assertRaises(ValueError):
                BselSunayama23(*args)
        model = BselSunayama23([1.2], [1.0], 0.1, [5])
        self.assertTrue(model.validate_wall([5]))
        with self.assertRaises(ValueError):
            model.validate_wall([2])
        with self.assertRaises(ValueError):
            model(-1.0, 5)
        with self.assertRaises(ValueError):
            model.derivative(0.0, 5)

    def test_sunayama_rejects_nonfinite_radius(self):
        model = BselSunayama23([1.2], [1.0], 0.1, [5])
        with self.assertRaises(ValueError):
            model(np.nan, 5)

    def test_sunayama_rejects_fractional_bin_label(self):
        # Labels are keys, so truncating 2.5 to 2 would select a wrong row.
        with self.assertRaises(ValueError):
            BselSunayama23([1.2], [1.0], 0.1, [2.5])
        model = BselSunayama23([1.2], [1.0], 0.1, [2])
        with self.assertRaises(ValueError):
            model(1.0, 2.5)
        with self.assertRaises(ValueError):
            model.derivative(1.0, 2.5)
        with self.assertRaises(ValueError):
            model.validate_wall([2.5])
        source = {
            ("fractional_bins", "pi0"): [1.2],
            ("fractional_bins", "r0"): [1.0],
            ("fractional_bins", "c"): 0.1,
            ("fractional_bins", "lambda_bin"): [2.5],
        }
        with self.assertRaises(ValueError):
            BselSunayama23.from_source(source, section="fractional_bins")

    def test_sunayama_rejects_duplicate_bin_labels(self):
        with self.assertRaises(ValueError):
            BselSunayama23([1.2, 1.4], [1.0, 2.0], 0.1, [2, 2])

    def test_datablock_loading_default_and_custom_sections(self):
        block = {
            ("bsel_profile_costanzi26", "A"): 0.1,
            ("bsel_profile_costanzi26", "alpha"): 0.92,
            ("bsel_profile_costanzi26", "beta"): -0.53,
            ("bsel_profile_costanzi26", "gamma"): 4.1,
            ("custom_sunayama", "pi0"): np.array([1.2, 1.4]),
            ("custom_sunayama", "r0"): np.array([1.0, 2.0]),
            ("custom_sunayama", "c"): 0.1,
            ("custom_sunayama", "lambda_bin"): np.array([2, 5]),
        }
        costanzi = BselModels.from_source(block, "Costanzi26")
        sunayama = BselModels.from_source(block, "Sunayama23",
                                           section="custom_sunayama")
        np.testing.assert_allclose(costanzi(1.0, bin_index=2, lob=40.0, z=0.3),
                                   _costanzi_decimal_reference(1.0)[0], rtol=2e-15)
        self.assertAlmostEqual(float(sunayama(4.0, bin_index=5)),
                               2.4 + 0.1 * np.log(2.0))
        with self.assertRaises(KeyError):
            BselModels.from_source(block, "Sunayama23")

    def test_dispatcher(self):
        costanzi = BselModels("Costanzi26", BselCostanzi26(0.1, 0.92, -0.53, 4.1))
        sunayama = BselModels("Sunayama23", BselSunayama23([1.2], [1.0], 0.1))
        self.assertGreater(float(costanzi(1.0, bin_index=0, lob=40.0, z=0.3)), 1.0)
        self.assertAlmostEqual(float(sunayama(0.5, bin_index=0)), 1.6)
        with self.assertRaises(ValueError):
            BselModels("unknown", BselSunayama23([1.2], [1.0], 0.1))
        with self.assertRaises(TypeError):
            BselModels("Costanzi26", BselSunayama23([1.2], [1.0], 0.1))
        with self.assertRaises(ValueError):
            costanzi(1.0, bin_index=0)
        with self.assertRaises(ValueError):
            costanzi.derivative(1.0, bin_index=0)
        np.testing.assert_allclose(sunayama.derivative(0.5, bin_index=0), 1.2)


if __name__ == "__main__":
    unittest.main()
