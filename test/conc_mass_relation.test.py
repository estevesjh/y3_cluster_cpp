#!/usr/bin/env python
"""Unit tests for the concentration-mass relations (issues #3, #13).

Owner-ratified convention (review 2026-08-20): the 1-halo term uses the
concentration evaluated at the cluster redshift (``one_halo_z``), and
the shearPrj projection profile uses the concentration-MASS relation
(``use_halo_model_conc``) instead of the legacy fixed c = 4.

External validation legs (references re-typed from the PAPERS, never
from the code under test):

* Child et al. 2018 (ApJ 859, 55), Eq. 19 + Table 2: the four-sample
  peak-height form c = c0 + A [ (M/(b M*))^m (1 + M/(b M*))^{-m} - 1 ]
  with M* the non-linear mass, log10 M*(z) = 12.5 - 1.5 z (their
  colossus/WMAP7 fit, haloModel.peakHeight_nonLinearMass docstring).
* Duffy et al. 2008 (MNRAS 390, L64), Table 1 full sample:
  c = A (M / 2e12 h^-1 Msun)^B (1+z)^C with (A, B, C) =
  (7.85, -0.081, -0.71).

Plus the conc plumbing added for #13:
* NfwDsigmaMisProduction(conc=4.0) must reproduce conc=None (the legacy
  fixed-c path) exactly -- guards the constants against drift.
* The published haloModel/concentration table must equal
  child18(M_grid, one_halo_z) -- wiring check on the fiducial dump
  (the committed dump uses the legacy default one_halo_z = 0).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "y3_buzzard"))
sys.path.insert(0, str(REPO / "src" / "pipelines"))

from haloModel import (child18_mass_concentration,          # noqa: E402
                       duffy_concentration_relation,
                       peakHeight_nonLinearMass)

DUMP_DIR = REPO / "cosmosis-models" / "real_pipeline_extract_prj2h_output"
HAS_DUMP = (DUMP_DIR / "halomodel").is_dir()

# Child+18 Table 2 (M200c), re-typed from the paper:
#   sample:              (m,     A,     b,       c0)
CHILD18_TABLE2 = {
    "individual_all":     (-0.10, 3.44, 430.49,  3.19),
    "individual_relaxed": (-0.09, 2.88, 1644.53, 3.54),
    "stacked_nfw":        (-0.07, 4.61, 638.65,  3.59),
    "stacked_einasto":    (-0.01, 63.2, 431.48,  3.36),
}
# Duffy+08 Table 1, full sample: c = A (M/Mpiv)^B (1+z)^C
DUFFY08 = dict(A=7.85, B=-0.081, C=-0.71, Mpiv=2.0e12)

# (lnM [Msun/h], z) sample points spanning the production range
POINTS = [(30.5, 0.0), (32.0, 0.275), (33.5, 0.46), (35.0, 0.575)]


def _child18_paper(lnM, z, sample):
    """Independent implementation straight from Child+18 Eq. 19."""
    m, A, b, c0 = CHILD18_TABLE2[sample]
    Mstar = 10.0 ** (12.5 - 1.5 * z)
    x = np.exp(lnM) / (Mstar * b)
    return c0 + A * (x**m * (1.0 + x) ** (-m) - 1.0)


def _duffy_paper(lnM, z):
    p = DUFFY08
    return p["A"] * (np.exp(lnM) / p["Mpiv"]) ** p["B"] * (1.0 + z) ** p["C"]


class TestChild18AgainstPaper(unittest.TestCase):
    def test_all_four_samples_match_eq19_table2(self):
        for sample in CHILD18_TABLE2:
            for lnM, z in POINTS:
                got = float(child18_mass_concentration(
                    np.array([np.exp(lnM)]), z, halo_sample=sample)[0])
                ref = _child18_paper(lnM, z, sample)
                self.assertAlmostEqual(got, ref, places=10,
                                       msg=f"{sample} at (lnM={lnM}, z={z})")

    def test_nonlinear_mass_is_the_documented_loglinear_fit(self):
        for z in (0.0, 0.275, 0.46, 1.0):
            self.assertAlmostEqual(np.log10(peakHeight_nonLinearMass(z)),
                                   12.5 - 1.5 * z, places=12)

    def test_stacked_nfw_monotone_decreasing_over_production_range(self):
        lnM = np.linspace(29.9336, 35.6814, 200)
        c = child18_mass_concentration(np.exp(lnM), 0.46,
                                       halo_sample="stacked_nfw")
        self.assertTrue(np.all(np.diff(c) < 0.0))
        self.assertTrue(np.all(c > 3.0) and np.all(c < 6.0))

    def test_concentration_decreases_with_redshift(self):
        M = np.exp(32.5)
        cs = [float(child18_mass_concentration(
            np.array([M]), z, halo_sample="stacked_nfw")[0])
            for z in (0.0, 0.275, 0.46, 0.65)]
        self.assertTrue(all(a > b for a, b in zip(cs, cs[1:])))


class TestDuffy08AgainstPaper(unittest.TestCase):
    def test_matches_table1_full_sample_power_law(self):
        for lnM, z in POINTS:
            got = float(duffy_concentration_relation(
                np.array([np.exp(lnM)]), z_eff=z)[0])
            ref = _duffy_paper(lnM, z)
            self.assertAlmostEqual(got, ref, places=10,
                                   msg=f"Duffy08 at (lnM={lnM}, z={z})")

    def test_tracks_child18_within_a_few_percent_where_it_matters(self):
        # The two relations must stay close over the production range
        # (the #13 figure's point: relation choice << the fixed-c=4
        # error). Guard the envelope so a bad edit to either shows up.
        lnM = np.linspace(29.9336, 35.6814, 50)
        cc = child18_mass_concentration(np.exp(lnM), 0.46,
                                        halo_sample="stacked_nfw")
        cd = duffy_concentration_relation(np.exp(lnM), z_eff=0.46)
        self.assertLess(np.abs(cd / cc - 1.0).max(), 0.12)


class TestConcPlumbing(unittest.TestCase):
    def test_explicit_conc4_reproduces_legacy_fixed_path_exactly(self):
        from shared.lensing_profiles import NfwDsigmaMisProduction, CONC
        dsmis = NfwDsigmaMisProduction(kernel="single")
        R = np.array([0.2, 1.0, 5.0])
        lnM = np.array([31.0, 33.0, 35.0])
        legacy = dsmis(R[:, None], 0.3, lnM[None, :], rho_mult=0.31)
        explicit = dsmis(R[:, None], 0.3, lnM[None, :], rho_mult=0.31,
                         conc=CONC)
        np.testing.assert_array_equal(explicit, legacy)

    def test_conc_scaling_follows_rs_and_delta_c_analytically(self):
        # The lookup is universal in x = r/r_s, so at fixed x the profile
        # must scale exactly as r_s(c) * delta_c(c). Query at r chosen so
        # both c's hit the SAME x (and same clamped x_mis), making the
        # table factor cancel identically.
        from shared.lensing_profiles import NfwDsigmaMisProduction, RHOC
        dsmis = NfwDsigmaMisProduction(kernel="single")
        lnM, c1, c2, x = 33.0, 4.0, 4.8, 0.7
        r200 = np.cbrt(3.0 * np.exp(lnM) / (800.0 * np.pi * RHOC))
        out = []
        for c in (c1, c2):
            r_s = r200 / c
            out.append(float(dsmis(np.array([x * r_s]), 1e-8, lnM,
                                   rho_mult=0.31, conc=c)[0]))
        dc = lambda c: (200.0 * c**3 / 3.0) / (np.log1p(c) - c / (1.0 + c))
        expected = (r200 / c2 * dc(c2)) / (r200 / c1 * dc(c1))
        self.assertAlmostEqual(out[1] / out[0], expected, places=10)

    @unittest.skipUnless(HAS_DUMP, f"requires the fiducial dump at {DUMP_DIR}")
    def test_published_concentration_table_is_child18_at_one_halo_z(self):
        # Committed dump = legacy default one_halo_z = 0.
        lnm = np.loadtxt(DUMP_DIR / "halomodel" / "lnm.txt")
        c_tab = np.loadtxt(DUMP_DIR / "halomodel" / "concentration.txt")
        ref = child18_mass_concentration(np.exp(lnm), 0.0,
                                         halo_sample="stacked_nfw")
        np.testing.assert_allclose(c_tab, ref, rtol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
