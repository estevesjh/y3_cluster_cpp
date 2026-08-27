#!/usr/bin/env python3
"""Convergence tests for ``src/pipelines/shared/explicit_grid_core.py``.

This module is the project's ACCURACY REFERENCE: CLAUDE.md's testing
policy says every accuracy number is quoted against
``explicit_mass_integral_adaptive`` ("reported error <= 1e-6"), never
against a production ``.so``. It had no test of its own convergence -- the
reference was the one thing in the suite nobody checked.

Three legs, all dump-free (``mor`` is a plain six-float dict and the
projection splines come from ``PrjParams.default()``, so nothing here needs
a CosmoSIS run):

1. The quadrature primitive ``datablock_models.gl_nodes``, on which every
   0d backend and this reference both rest: exact on polynomials up to
   degree 2n-1, and a lognormal x power-law toy -- an integrand shaped like
   the real mass integrand, with no closed form -- against
   ``scipy.integrate.quad`` to 1e-12.

2. The adaptive integrator against ``scipy.integrate.quad`` (QUADPACK), a
   genuinely independent adaptive scheme with its own rigorous error
   estimate, via the module's own ``explicit_mass_integral_quad``. This is
   what actually substantiates the "reported error <= 1e-6" claim: the
   reported error must both be <= epsrel AND actually bound the deviation
   from the independent integrator.

3. The documented exact identity between the z-contracted and z-resolved
   weight builders ("Summing W2d over q reproduces explicit_mass_weights
   exactly"), and the profile-hook identity (a unit profile must reproduce
   the counts path).
"""
from __future__ import annotations

import sys
import types
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import quad

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))


def _install_cosmosis_stub():
    """Minimal ``cosmosis.datablock`` so ``sel_function`` imports.

    ``sel_function.py`` does ``from cosmosis.datablock import
    option_section`` at module scope, and ``sel_kernels.load()`` -- which
    ``explicit_grid_core`` calls -- executes it. CosmoSIS's Python package
    is not importable outside a CosmoSIS environment, and none of the
    quadrature under test needs it. Installed before the first import
    below, not in a setUpClass, so it does not depend on test ordering.
    """
    if "cosmosis.datablock" in sys.modules:
        return
    stub = types.ModuleType("cosmosis")
    datablock = types.ModuleType("cosmosis.datablock")
    datablock.option_section = "module_options"
    datablock.names = types.SimpleNamespace(
        cosmological_parameters="cosmological_parameters",
        distances="distances")
    stub.datablock = datablock
    sys.modules.setdefault("cosmosis", stub)
    sys.modules.setdefault("cosmosis.datablock", datablock)


_install_cosmosis_stub()

from shared import datablock_models as dm            # noqa: E402
from shared import explicit_grid_core as egc         # noqa: E402
from systematics.selection_richness.python import sel_kernels  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# A fiducial-ish MOR: plain floats, no datablock needed.
MOR = {
    "log10_Mmin": 12.3,
    "log10_M1": 13.4,
    "alpha": 1.0,
    "epsilon": 0.0,
    "sigma_lambda": 0.25,
    "z_pivot": 0.45,
}

# Two DES-Y3-shaped bins.  Two is enough to exercise per-bin mass limits
# and the shared edge array while keeping the scipy leg affordable.
BINS = {
    "lam_min": np.array([20.0, 45.0]),
    "lam_max": np.array([30.0, 60.0]),
    "zob_min": np.array([0.20, 0.20]),
    "zob_max": np.array([0.35, 0.35]),
    "sigma_z": np.array([0.02, 0.02]),
}

WINDOW = dict(zt_low=0.10, zt_high=0.55, lnm_low=29.9336, lnm_high=36.73)


def _hmf(ln_mass, z):
    """A smooth positive stand-in dn/dlnM(lnM, z), broadcasting in both."""
    return 1.0e-4 * np.exp(-0.9 * (ln_mass - 30.0)) / (1.0 + z) ** 2


def _dv(z):
    """A smooth positive stand-in dV/dOmega/dz(z)."""
    return 1.0e9 * np.asarray(z) ** 2 / (1.0 + np.asarray(z))


def _sci(z):
    """A smooth positive stand-in Sigma_crit^-1(z)."""
    return 3.0e-4 * (1.0 + 0.2 * np.asarray(z))


def _splines():
    return sel_kernels.plob_splines_default()


class TestGaussLegendrePrimitive(unittest.TestCase):
    """``dm.gl_nodes`` -- the quadrature every 0d backend is built on."""

    def test_exact_on_polynomials_up_to_degree_2n_minus_1(self):
        a, b = -0.7, 1.4
        for n in (2, 3, 5, 8, 16):
            x, w = dm.gl_nodes(a, b, n)
            self.assertEqual(x.size, n)
            self.assertEqual(w.size, n)
            for p in range(2 * n):
                got = float(w @ (x ** p))
                exact = (b ** (p + 1) - a ** (p + 1)) / (p + 1)
                self.assertLess(abs(got - exact), 1e-11 * max(1.0, abs(exact)),
                                f"n={n} degree={p}")

    def test_weights_sum_to_the_interval_and_nodes_are_interior(self):
        a, b = 0.13, 4.7
        for n in (2, 4, 12, 32):
            x, w = dm.gl_nodes(a, b, n)
            self.assertAlmostEqual(float(w.sum()) / (b - a), 1.0, places=13)
            self.assertTrue(np.all(x > a))
            self.assertTrue(np.all(x < b))
            self.assertTrue(np.all(np.diff(x) > 0.0))
            # Symmetric about the midpoint.
            self.assertTrue(np.allclose(x + x[::-1], a + b, atol=1e-12))

    def test_lognormal_times_power_law_toy_matches_scipy_quad(self):
        # An integrand shaped like the real mass integrand -- a peaked
        # lognormal in mass times a power law -- with no closed form, so
        # the reference is scipy's QUADPACK at machine-level tolerance.
        mu, sigma, slope = 33.0, 0.55, -1.3

        def integrand(ln_mass):
            return (np.exp(-0.5 * ((ln_mass - mu) / sigma) ** 2)
                    * np.exp(slope * (ln_mass - mu)))

        lo, hi = 29.9336, 36.73
        exact, err = quad(integrand, lo, hi, epsabs=0.0, epsrel=1e-13,
                          limit=400)
        self.assertLess(err / exact, 1e-11, "scipy reference is not tight")

        x, w = dm.gl_nodes(lo, hi, 96)
        got = float(w @ integrand(x))
        self.assertLess(abs(got / exact - 1.0), 1e-12,
                        "GL-96 must resolve this integrand to 1e-12")

        # ...and a deliberately under-resolved rule must NOT: otherwise the
        # check above would pass for any implementation, correct or not.
        x8, w8 = dm.gl_nodes(lo, hi, 8)
        self.assertGreater(abs(float(w8 @ integrand(x8)) / exact - 1.0), 1e-9)


class TestAdaptiveReferenceConvergence(unittest.TestCase):
    """The claim CLAUDE.md's accuracy policy rests on."""

    @classmethod
    def setUpClass(cls):
        cls.splines = _splines()

    def _adaptive(self, epsrel, **extra):
        return egc.explicit_mass_integral_adaptive(
            BINS, MOR, self.splines, _hmf, _dv, epsrel=epsrel,
            **WINDOW, **extra)

    def test_reported_error_respects_the_requested_epsrel(self):
        for epsrel in (1e-4, 1e-6):
            vals, errs = self._adaptive(epsrel)
            self.assertTrue(np.all(np.isfinite(vals)))
            self.assertTrue(np.all(vals > 0.0))
            rel = errs / np.abs(vals)
            self.assertTrue(np.all(rel <= epsrel),
                            f"epsrel={epsrel}: reported {rel.max():.2e}")

    def test_reported_error_actually_bounds_the_deviation_from_scipy_quad(self):
        # The substantive check: `errs` is an embedded GL-10/GL-20
        # difference, which is an ESTIMATE.  Comparing against
        # scipy.integrate.quad -- a different adaptive scheme with its own
        # rigorous bound -- tests whether that estimate is honest.
        vals, errs = self._adaptive(1e-6)
        ref, ref_err = egc.explicit_mass_integral_quad(
            BINS, MOR, self.splines, _hmf, _dv, epsrel=1e-9, **WINDOW)
        self.assertTrue(np.all(ref_err / np.abs(ref) < 1e-7),
                        "the scipy reference is not tight enough to judge")
        dev = np.abs(vals / ref - 1.0)
        self.assertTrue(np.all(dev < 1e-6),
                        f"adaptive vs quad: {dev.max():.2e}")
        # ...and the reported error is not an underestimate of it.
        self.assertTrue(np.all(errs / np.abs(vals) + 1e-12 >= 0.0))
        self.assertTrue(np.all(dev <= np.maximum(errs / np.abs(vals), 1e-9)),
                        f"reported error underestimates the true deviation: "
                        f"dev={dev}, reported={errs / np.abs(vals)}")

    def test_tightening_epsrel_moves_the_answer_by_less_than_the_loose_bound(self):
        loose_vals, loose_errs = self._adaptive(1e-3)
        tight_vals, _ = self._adaptive(1e-7)
        moved = np.abs(loose_vals / tight_vals - 1.0)
        self.assertTrue(np.all(moved <= loose_errs / np.abs(loose_vals)),
                        f"moved {moved} beyond the loose bound "
                        f"{loose_errs / np.abs(loose_vals)}")

    def test_sigma_crit_inv_enters_the_shear_variant_multiplicatively(self):
        # sci is folded into the z contraction, so a CONSTANT sci must
        # scale the result by exactly that constant -- a check that the
        # optional hook is wired into the weight and not, say, applied
        # twice or dropped.
        base, _ = self._adaptive(1e-8)
        scaled, _ = self._adaptive(1e-8, sci=lambda z: 2.0 * np.ones_like(
            np.asarray(z, dtype=float)))
        self.assertTrue(np.allclose(scaled, 2.0 * base, rtol=1e-6))


class TestWeightBuilderIdentities(unittest.TestCase):
    """Exact identities the module's own docstrings promise."""

    @classmethod
    def setUpClass(cls):
        cls.splines = _splines()

    def test_z_resolved_weights_sum_to_the_z_contracted_ones(self):
        # explicit_mass_z_weights' docstring: "Summing W2d over q
        # reproduces explicit_mass_weights exactly."  The max-model backend
        # depends on this being exact, not approximate.
        lnm_x, lnm_w, weights = egc.explicit_mass_weights(
            BINS, MOR, self.splines, _hmf, _dv, n_lnm=24, n_z=32, **WINDOW)
        lnm_x2, lnm_w2, _z_x, w2d = egc.explicit_mass_z_weights(
            BINS, MOR, self.splines, _hmf, _dv, n_lnm=24, n_z=32, **WINDOW)

        self.assertTrue(np.array_equal(lnm_x, lnm_x2))
        self.assertTrue(np.array_equal(lnm_w, lnm_w2))
        self.assertTrue(np.allclose(w2d.sum(axis=-1), weights,
                                    rtol=1e-14, atol=0.0))

    def test_sigma_crit_inv_agrees_between_the_two_builders(self):
        _x, _w, weights = egc.explicit_mass_weights(
            BINS, MOR, self.splines, _hmf, _dv, sci=_sci, n_lnm=24, n_z=32,
            **WINDOW)
        _x2, _w2, _z, w2d = egc.explicit_mass_z_weights(
            BINS, MOR, self.splines, _hmf, _dv, sci=_sci, n_lnm=24, n_z=32,
            **WINDOW)
        self.assertTrue(np.allclose(w2d.sum(axis=-1), weights,
                                    rtol=1e-14, atol=0.0))

    def test_weights_are_positive_and_bin_ordering_is_preserved(self):
        _x, _w, weights = egc.explicit_mass_weights(
            BINS, MOR, self.splines, _hmf, _dv, n_lnm=24, n_z=32, **WINDOW)
        self.assertEqual(weights.shape, (2, 24))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= 0.0))
        # The low-richness bin must collect more clusters than the
        # high-richness one at every mass node's aggregate -- if the bin
        # index were transposed this would flip.
        self.assertGreater(weights[0].sum(), weights[1].sum())

    def test_unit_profile_reproduces_the_counts_path(self):
        # profile(b, R, lnM) == 1 turns the shear branch into the counts
        # branch, so the two code paths must return the same number for
        # every radius.  This pins the profile hook's argument order too:
        # a (R, lnM) swap would not survive the broadcast.
        counts, _ = egc.explicit_mass_integral_adaptive(
            BINS, MOR, self.splines, _hmf, _dv, epsrel=1e-8, **WINDOW)
        radii = np.array([0.5, 1.0, 2.0])
        shear, _ = egc.explicit_mass_integral_adaptive(
            BINS, MOR, self.splines, _hmf, _dv, epsrel=1e-8,
            profile=lambda b, r, m: np.ones_like(r * m),
            r_perp=radii, **WINDOW)
        self.assertEqual(shear.shape, (2, radii.size))
        for i in range(radii.size):
            self.assertTrue(np.allclose(shear[:, i], counts, rtol=1e-6),
                            f"radius index {i}")

    def test_profile_hook_receives_radius_and_mass_in_the_documented_order(self):
        # profile(b, R, lnM): R has shape (n_r, 1) and lnM (1, n_m).  A
        # backend that swapped them would still broadcast, so this is
        # checked directly rather than through the output.
        seen = []

        def profile(b, r, m):
            seen.append((np.shape(r), np.shape(m)))
            return np.ones_like(r * m)

        radii = np.array([0.5, 1.0, 2.0])
        egc.explicit_mass_integral_adaptive(
            BINS, MOR, self.splines, _hmf, _dv, epsrel=1e-3,
            profile=profile, r_perp=radii, **WINDOW)
        self.assertTrue(seen, "profile hook was never called")
        for r_shape, m_shape in seen:
            self.assertEqual(r_shape, (radii.size, 1))
            self.assertEqual(len(m_shape), 2)
            self.assertEqual(m_shape[0], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
