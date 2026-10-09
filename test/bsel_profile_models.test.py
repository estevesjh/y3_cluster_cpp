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

from systematics.selection_boost.Bsel import (  # noqa: E402
    BselModels,
    Shear1h2hMaxSelCostanzi26,
    Shear1h2hMaxSelSunayama23,
    selected_shear1h2h_max_consumer,
)
from systematics.selection_boost.BselCostanzi26 import (  # noqa: E402
    BselCostanzi26,
    SECTION as COSTANZI_SECTION,
)
from systematics.selection_boost.BselSunayama23 import BselSunayama23  # noqa: E402


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

    def test_ten_radius_benchmark(self):
        # Independent benchmark across R=logspace(-1, 1, 10).  The
        # Sunayama values cross its R0=3.5 branch in the same sample.
        radii = np.geomspace(0.1, 10.0, 10)
        costanzi = BselCostanzi26(0.10, 0.92, -0.53, 4.1)
        costanzi_values = np.array([
            1.0111785186792333, 1.01789639910452, 1.0286218236572273,
            1.0453968273535741, 1.0678416280469756, 1.0791840601769058,
            1.0680253815140701, 1.0528016345802402, 1.0403504761945139,
            1.0307745323736017])
        costanzi_derivatives = np.array([
            0.10283306354875656, 0.09863043480487986,
            0.09406590730077931, 0.08570538684808919,
            0.05493575001600590, -0.00347956712456455,
            -0.01416523906054845, -0.00763264961463009,
            -0.00355862842257145, -0.00163056002101080])
        np.testing.assert_allclose(costanzi(radii, 40.0, 0.3),
                                   costanzi_values, rtol=2e-14, atol=1e-15)
        np.testing.assert_allclose(costanzi.derivative(radii, 40.0, 0.3),
                                   costanzi_derivatives, rtol=2e-13, atol=1e-15)

        sunayama = BselSunayama23([0.45], [3.5], -0.1, [0])
        sunayama_values = np.array([
            1.0128571428571429, 1.021447006906858, 1.035775763742663,
            1.0596775707178785, 1.0995481877732878, 1.1660563855019137,
            1.2769987458612422, 1.447354902794312, 1.3961863451722221,
            1.3450177875501321])
        sunayama_derivatives = np.array([
            0.12857142857142859, 0.12857142857142859,
            0.12857142857142859, 0.12857142857142859,
            0.12857142857142859, 0.12857142857142859,
            0.12857142857142859, -0.02782559402207126,
            -0.01668100537200059, -0.01])
        np.testing.assert_allclose(sunayama(radii, 0), sunayama_values,
                                   rtol=2e-14, atol=1e-15)
        np.testing.assert_allclose(sunayama.derivative(radii, 0),
                                   sunayama_derivatives, rtol=2e-14,
                                   atol=1e-15)

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
            ("boost_selection_costanzi26", "A"): 0.1,
            ("boost_selection_costanzi26", "alpha"): 0.92,
            ("boost_selection_costanzi26", "beta"): -0.53,
            ("boost_selection_costanzi26", "gamma"): 4.1,
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

    def test_costanzi_default_section_is_the_selection_boost_section(self):
        self.assertEqual(COSTANZI_SECTION, "boost_selection_costanzi26")

    def test_selected_consumers_preserve_the_local_profile_shape(self):
        profile = np.array([[2.0, 3.0], [5.0, 7.0]])
        costanzi = Shear1h2hMaxSelCostanzi26(
            BselCostanzi26(0.1, 0.92, -0.53, 4.1))
        sunayama = Shear1h2hMaxSelSunayama23(
            BselSunayama23([0.45], [3.5], -0.1, [4]))
        costanzi_out = costanzi(profile, 1.0, bin_index=4, lob=40.0, z=0.3)
        sunayama_out = sunayama(profile, 1.0, bin_index=4)
        np.testing.assert_allclose(
            costanzi_out, profile * BselCostanzi26(0.1, 0.92, -0.53, 4.1)(1.0, 40.0, 0.3))
        np.testing.assert_allclose(
            sunayama_out, profile * BselSunayama23([0.45], [3.5], -0.1, [4])(1.0, 4))

    def test_selected_consumer_factory_loads_both_tags(self):
        block = {
            ("boost_selection_costanzi26", "A"): 0.1,
            ("boost_selection_costanzi26", "alpha"): 0.92,
            ("boost_selection_costanzi26", "beta"): -0.53,
            ("boost_selection_costanzi26", "gamma"): 4.1,
            ("boost_selection_sunayama", "pi0"): [0.45],
            ("boost_selection_sunayama", "r0"): [3.5],
            ("boost_selection_sunayama", "c"): -0.1,
            ("boost_selection_sunayama", "lambda_bin"): [4],
        }
        self.assertIsInstance(selected_shear1h2h_max_consumer(
            block, "Costanzi26"), Shear1h2hMaxSelCostanzi26)
        self.assertIsInstance(selected_shear1h2h_max_consumer(
            block, "Sunayama23", "boost_selection_sunayama"),
            Shear1h2hMaxSelSunayama23)

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
