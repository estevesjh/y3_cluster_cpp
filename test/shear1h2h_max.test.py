#!/usr/bin/env python3
"""Unit tests for ``src/pipelines/des_y3/shear_1h2h/python/0d/shear1h2h_max.py``,
the Python twin of the C++ ``Shear1h2hMax`` (shear1h2h_max_t.hh).

1. Option contract of the ``Shear1h2hMax`` class (self-contained).
2. Real-pipeline parity (skipped unless the macOS/NERSC cosmosis env and the
   built ``Shear1h2hMax.so`` are available): ``des_y3_python0d_fast_apriori.ini``
   vs ``des_y3_cpp0d_fast_apriori.ini`` at the fiducial point, for
   profile_method = piecewise_approx and gauss_legendre:
     - Python == C++ (piecewise to 1e-12; GL to 1e-6, scipy vs GSL bicubic);
     - GL default nodes (64/32) vs doubled (128/64) within 1e-4;
     - piecewise_approx vs gauss_legendre within 0.5% (the approximation).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PIPELINES = REPO / "src" / "pipelines"
MODULE_DIR = PIPELINES / "des_y3" / "shear_1h2h" / "python" / "0d"
sys.path.insert(0, str(PIPELINES))
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import shear1h2h_max as mod                              # noqa: E402
from _dump_datablock import make_options                # noqa: E402

ENVELOPE = dict(zt_low=0.05, zt_high=0.80, lnm_low=29.9336, lnm_high=36.7300)
R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
BIN_INDEX = np.arange(12)

CPP_SO = (REPO / "release-build" / "src" / "modules" / "des_y3_shear1h_0d_cpp"
          / "Shear1h2hMax.so")
HAS_PIPELINE = (CPP_SO.is_file() and bool(os.environ.get("CAMB_EMULATOR_DIR"))
                and bool(os.environ.get("COSMOSIS_STANDARD_LIBRARY")))
_SKIP_MSG = ("requires the cosmosis env (cosmosis_init_macos.sh) and a built "
             f"{CPP_SO}")
UPSTREAM = ("consistency GrowthFactor cp_camb MfTinker halo_model "
            "average_sigma_crit_inv sel_function")


def _base_options():
    return dict(ENVELOPE, bin_index=BIN_INDEX, r_perp=R_PERP)


def _run_pipeline_block(backend, **overrides):
    """Run the fast_apriori ini (python or cpp module) at the fiducial point."""
    from cosmosis.runtime.config import Inifile
    from cosmosis.runtime.pipeline import LikelihoodPipeline
    if backend == "python":
        ini_name, sec = "des_y3_python0d_fast_apriori.ini", "shear1h2h_max"
    else:
        ini_name, sec = "des_y3_cpp0d_fast_apriori.ini", "Shear1h2hMax"
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        ini = Inifile(str(REPO / "cosmosis-models" / ini_name))
        ini.set("pipeline", "modules", f"{UPSTREAM} {sec}")
        ini.set("pipeline", "quiet", "T")
        for key, value in overrides.items():
            ini.set(sec, key, str(value))
        pipe = LikelihoodPipeline(ini)
        return pipe.run_parameters(pipe.start_vector())
    finally:
        os.chdir(cwd)


class TestOptionContract(unittest.TestCase):
    def test_defaults(self):
        m = mod.setup(make_options(_base_options()))
        self.assertIsInstance(m, mod.Shear1h2hMax)
        self.assertEqual((m.n_lnm, m.n_z), (96, 64))
        self.assertEqual(m.profile_method, "piecewise_approx")
        self.assertEqual((m.n_los, m.n_aperture), (64, 32))
        self.assertFalse(m.bsel_enabled)
        np.testing.assert_array_equal(m.bin_index, BIN_INDEX)
        np.testing.assert_allclose(m.r_perp, R_PERP)

    def test_overrides(self):
        m = mod.setup(make_options(dict(
            _base_options(), n_lnm=48, n_z=32, profile_method="gauss_legendre",
            n_los=128, n_aperture=64, bsel=True, bsel_model="Costanzi26")))
        self.assertEqual((m.n_lnm, m.n_z), (48, 32))
        self.assertEqual(m.profile_method, "gauss_legendre")
        self.assertEqual((m.n_los, m.n_aperture), (128, 64))
        self.assertTrue(m.bsel_enabled)
        self.assertEqual(m.bsel_model, "Costanzi26")

    def test_unknown_profile_method_raises(self):
        with self.assertRaises(ValueError):
            mod.setup(make_options(dict(_base_options(), profile_method="bad")))

    def test_missing_bin_index_or_r_perp_fails_loudly(self):
        for missing in ("bin_index", "r_perp"):
            entries = {k: v for k, v in _base_options().items() if k != missing}
            with self.assertRaises(Exception, msg=f"missing {missing}"):
                mod.setup(make_options(entries))

    def test_cleanup_is_a_no_op(self):
        self.assertEqual(mod.cleanup({}), 0)


@unittest.skipUnless(HAS_PIPELINE, _SKIP_MSG)
class TestPythonMatchesCpp(unittest.TestCase):
    """Same ini envelope, same fiducial point, both backends."""

    @staticmethod
    def _run(backend, **overrides):
        block = _run_pipeline_block(backend, **overrides)
        return np.array(block["shear1h2h_max", "vals"]).reshape(12, 10)

    @classmethod
    def setUpClass(cls):
        cls.py_pw = cls._run("python", profile_method="piecewise_approx")
        cls.cpp_pw = cls._run("cpp", profile_method="piecewise_approx")
        cls.py_gl = cls._run("python", profile_method="gauss_legendre")
        cls.cpp_gl = cls._run("cpp", profile_method="gauss_legendre")
        cls.py_gl2 = cls._run("python", profile_method="gauss_legendre",
                              n_los=128, n_aperture=64)

    def test_piecewise_python_equals_cpp(self):
        np.testing.assert_allclose(self.py_pw, self.cpp_pw, rtol=1e-12, atol=0)

    def test_gauss_legendre_python_equals_cpp(self):
        # measured 1.0e-7: scipy RectBivariateSpline vs GSL bicubic for xi_nl.
        np.testing.assert_allclose(self.py_gl, self.cpp_gl, rtol=1e-6, atol=0)

    def test_gauss_legendre_default_vs_doubled_nodes(self):
        # measured 5.8e-5.
        np.testing.assert_allclose(self.py_gl, self.py_gl2, rtol=1e-4, atol=0)

    def test_piecewise_within_half_percent_of_gauss_legendre(self):
        # measured 4.3e-3 (piecewise low at R = 1.7-5 Mpc/h).
        np.testing.assert_allclose(self.py_pw, self.py_gl2, rtol=5e-3, atol=0)


@unittest.skipUnless(HAS_PIPELINE, _SKIP_MSG)
class TestGaussLegendreVsAdaptiveQuad(unittest.TestCase):
    """OperatorMax1h2h_gl against an independent adaptive integral.

    Real pipeline tables (halomodel bias/concentration/rho_m_ref, xi_nl read
    bicubic -- the same inputs the module feeds the operator).  Reference:
    scipy.integrate.quad for Sigma(R) along the line of sight with a
    breakpoint at the 1h/2h crossing, and for barSigma(<R) = 2/R^2 int R'
    Sigma(R') dR' (epsrel 1e-8 / 1e-7).  Per halo, no stacking, so this is
    the strict per-profile accuracy (mixed-sign errors partly cancel in
    the stack, where 64/32 is within 5.8e-5 of doubled nodes).
    Measured: 64/32 -> 1.5e-4, 128/64 -> 3.4e-5 (2 halos x 3 radii).
    """

    R = np.array([0.3, 1.71, 5.0])
    HALOS = ((32.0, 0.3), (35.0, 0.3))

    @classmethod
    def setUpClass(cls):
        from scipy.integrate import quad
        from scipy.optimize import brentq
        from cosmology.operator_max1h2h_gl import OperatorMax1h2h_gl
        from shared import datablock_models as dm
        cls.OperatorMax1h2h_gl = OperatorMax1h2h_gl

        block = _run_pipeline_block("python", profile_method="gauss_legendre")
        s = dm.DataBlockSource(block)
        rho_m = s.scalar("halomodel", "rho_m_ref")
        lnm_c, conc = s.array("halomodel", "lnm"), s.array("halomodel", "concentration")

        def xi_nfw(r, lnM):
            c = np.interp(lnM, lnm_c, conc)
            r200 = np.cbrt(3.0 * np.exp(lnM) / (800.0 * np.pi * rho_m))
            x = r * c / r200
            return ((200.0 / 3.0) * c ** 3 / (np.log1p(c) - c / (1.0 + c))
                    / (x * (1.0 + x) ** 2) - 1.0)

        bias = dm.Bilinear2D(s, "halomodel", "lnm", "z", "bias")
        xi_nl = dm.Bilinear2D(s, "xi_nl", "r", "z", "xi_nl", cubic=True)
        r_max = s.array("xi_nl", "r")[-1]
        cls.inputs = (xi_nfw, bias, xi_nl, rho_m, r_max)

        def delta_sigma_quad(R, lnM, z):
            b = float(np.squeeze(bias(lnM, z)))
            xi_2h = lambda r: b * float(np.squeeze(xi_nl(r, z)))
            r_t = brentq(lambda r: xi_nfw(r, lnM) - xi_2h(r), 0.05, 50.0)

            def sigma(Rp):
                f = lambda l: max(xi_nfw(np.hypot(Rp, l), lnM), xi_2h(np.hypot(Rp, l)))
                edges = ([0.0] + ([np.sqrt(r_t ** 2 - Rp ** 2)] if r_t > Rp else [])
                         + [10.0, np.sqrt(r_max ** 2 - Rp ** 2)])
                return 2.0 * rho_m * 1e-12 * sum(
                    quad(f, a, c, limit=200, epsabs=0, epsrel=1e-8)[0]
                    for a, c in zip(edges[:-1], edges[1:]))

            edges = [0.0] + ([r_t] if r_t < R else []) + [R]
            mean = 2.0 / R ** 2 * sum(
                quad(lambda x: x * sigma(x), a, c, limit=100, epsabs=0, epsrel=1e-7)[0]
                for a, c in zip(edges[:-1], edges[1:]))
            return mean - sigma(R)

        cls.ref = {h: np.array([delta_sigma_quad(r, *h) for r in cls.R])
                   for h in cls.HALOS}

    def _max_rel(self, n_los, n_aperture):
        op = self.OperatorMax1h2h_gl(*self.inputs, n_los=n_los, n_aperture=n_aperture)
        return max(np.abs(op.excess_surface_density(self.R, [m], [z])[:, 0, 0]
                          / self.ref[(m, z)] - 1.0).max() for m, z in self.HALOS)

    def test_default_nodes_within_2e_4_of_quad(self):
        self.assertLess(self._max_rel(64, 32), 2e-4)      # measured 1.5e-4

    def test_doubled_nodes_within_1e_4_of_quad(self):
        self.assertLess(self._max_rel(128, 64), 1e-4)     # measured 3.4e-5


if __name__ == "__main__":
    unittest.main(verbosity=2)
