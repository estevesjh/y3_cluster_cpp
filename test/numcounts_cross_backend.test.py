#!/usr/bin/env python3
"""Cross-backend / cross-strategy consistency for DES Y3 number counts.

Verifies that every implemented (strategy, backend) cell for the number
counts observable agrees with every other cell, per the matrix in
docs/source/pipeline_organization.md:

    full_ltmz  -> Python, C++, CUDA   (explicit-selection accuracy references)
    fast_mass  -> Python, C++ (production `NumCountsSel.so`, by identity)

The reference is the C++ `full_ltmz` backend (`NumCounts3d.so`,
adaptive Cuhre, eps_rel=1e-4) -- the project's designated accuracy
reference (docs/source/pipeline_organization.md "Validation policy").
Its values, and the CUDA `full_ltmz` backend's (`NumCounts3dGpu.so`,
PAGANI), are hard-coded below: reproducing them needs a full CosmoSIS
pipeline run of the compiled .so modules (and, for CUDA, a GPU), which
this lightweight, portable unittest does not attempt. They were
generated (2026-08-12, SLURM job 56780752 companion run) from a single
pipeline execution -- `gpu_smoke.ini`, modules `consistency prj_params
GrowthFactor cp_camb MfTinker halo_model average_sigma_crit_inv
sel_function NumCountsSel numcounts_explicit_gl NumCounts3d
NumCounts3dGpu` -- with the SAME `mock_mcmc_widePlanck_values.ini`
fiducial point and the SAME pinned 12-bin wall used everywhere else in
this repo's tests.

REGENERATED 2026-11-01 at sigma_z = 0.01 (the real-data cluster photo-z
width; previously 0.03): HEAD 24454a9 + the composite true-z rule patch
(the 3d reference modules are untouched by it), values from
des-nersc-cluster-scripts@main cosmosis-models/
mock_mcmc_widePlanck_values.ini, `prj_params` (table = hod) inserted
before sel_function, NumCounts3d eps_rel 1e-4 (status 0, err/val <=
1e-4), NumCounts3dGpu (PAGANI) on an A100. The previous sigma_z = 0.03
pins had gone stale at 9f6579e (HOD central counted once) unnoticed,
because the fiducial dump below was absent and every test here skipped.

The Python `full_ltmz` and `fast_mass` backends, and production's own
saved output, are all computed/read LIVE from this repo's checked-in
fiducial dump (cosmosis-models/real_pipeline_extract_output) -- verified
(2026-08-13) to reproduce the external pipeline run's Python full_ltmz
numbers to ~1e-7 relative, confirming the checked-in dump alone is a
sufficient, portable substitute for the full external run. Only the
two backends that cannot be cheaply re-invoked from a plain Python
process (they need real .so modules through a full CosmoSIS pipeline)
are pinned as literals.

To regenerate CPP_FULL_LTMZ / CUDA_FULL_LTMZ after a real change to
either backend: run a pipeline with `NumCounts3d`/
`NumCounts3dGpu` appended to `cosmosis-models/real_pipeline_extract.ini`
(same bin wall, `zt_low=0.05, zt_high=0.80, lnm_low=29.9336,
lnm_high=36.7300`, `lt_low`/`lt_high` per bin =
{0.1, 4*lam_max}) and read back `numcounts3d(gpu)/vals`.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))
sys.path.insert(0, str(REPO / "src" / "pipelines" / "des_y3"
                       / "number_counts" / "python" / "0d"))

from shared import datablock_models as dm  # noqa: E402
from systematics.selection_richness.python import sel_kernels  # noqa: E402
from numcounts_explicit_gl import compute_counts  # noqa: E402
from shared import explicit_grid_core  # noqa: E402

DUMP_DIR = REPO / "cosmosis-models" / "real_pipeline_extract_output"
HAS_DUMP = (DUMP_DIR / "matter_power_lin" / "p_k.txt").is_file()
_SKIP_MSG = (f"requires a real-pipeline dump at {DUMP_DIR} -- run "
            "`cosmosis cosmosis-models/real_pipeline_extract.ini` first")

# The pinned 12-bin wall (mock_mcmc_buzzard.ini sel_function section),
# identical to validate_against_fiducial.py / validate_vs_production.py.
BINS = dict(
    lam_min=np.array([20., 30., 45., 60.] * 3),
    lam_max=np.array([30., 45., 60., 200.] * 3),
    zob_min=np.array([0.20] * 4 + [0.35] * 4 + [0.50] * 4),
    zob_max=np.array([0.35] * 4 + [0.50] * 4 + [0.65] * 4),
    sigma_z=np.full(12, 0.01),
)
ZT_LOW, ZT_HIGH = 0.05, 0.80
LNM_LOW, LNM_HIGH = 29.9336, 36.7300

# C++ full_ltmz (NumCounts3d.so) -- THE reference. See module
# docstring for provenance.
CPP_FULL_LTMZ = np.array([
    1163.6413440132076, 459.20107804641225,
    130.70065321690086, 86.87107544220129,
    1905.922186969726, 691.5667959768751,
    178.93651433882016, 102.8028575532589,
    2078.884397405159, 698.3606459218564,
    165.16391848694374, 83.3647267365038,
])

# CUDA full_ltmz (NumCounts3dGpu.so, PAGANI).
CUDA_FULL_LTMZ = np.array([
    1163.6531838182468, 459.2016159941742,
    130.70140604769713, 86.87269550954336,
    1905.9386145242515, 691.5621836886763,
    178.9376035206231, 102.80371037242975,
    2078.901717303994, 698.3538779249859,
    165.16416569488882, 83.36590816665938,
])


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestNumberCountsCrossBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = dm.DumpSource(str(DUMP_DIR))
        cls.mor = sel_kernels.mor_from_source(cls.source)
        cls.plob = sel_kernels.plob_splines_default()
        cls.hmf = dm.HMF(cls.source)
        cls.dv = dm.DVDoDz(cls.source)

    def _fast_mass_python(self):
        weights = dm.MassZWeights(
            self.source, n_lnm=96, n_z=64, zt_lo=ZT_LOW, zt_hi=ZT_HIGH,
            lnm_lo=LNM_LOW, lnm_hi=LNM_HIGH, include_sci=False)
        return weights.norm()

    def test_python_full_ltmz_matches_cpp_full_ltmz(self):
        # Two completely different quadrature strategies (fixed-GL vs
        # adaptive Cuhre) over the same physics -- measured 8.6e-5.
        vals = compute_counts(BINS, self.mor, self.plob, self.hmf, self.dv,
                              zt_low=ZT_LOW, zt_high=ZT_HIGH,
                              lnm_low=LNM_LOW, lnm_high=LNM_HIGH)
        np.testing.assert_allclose(vals, CPP_FULL_LTMZ, rtol=3e-4)

    def test_cuda_full_ltmz_matches_cpp_full_ltmz(self):
        # Two independent adaptive integrators (PAGANI vs Cuhre) over the
        # identical integrand -- measured 1.9e-5.
        np.testing.assert_allclose(CUDA_FULL_LTMZ, CPP_FULL_LTMZ, rtol=1e-4)

    def test_python_fast_mass_matches_cpp_full_ltmz(self):
        # fast_mass (composite true-z rule, spline S_rich) vs the explicit
        # triple integral -- measured 3.2e-4. The old single 64-node z
        # panel on the bilinear S_stack table misses by 5.4e-3 here.
        np.testing.assert_allclose(self._fast_mass_python(), CPP_FULL_LTMZ,
                                   rtol=1e-3)

    def test_production_fast_mass_matches_cpp_full_ltmz(self):
        # Production NumCountsSel.so's own saved output vs full_ltmz --
        # measured 3.2e-4 (identical to the Python fast_mass check).
        vals = self.source.array("numcountssel", "vals")
        np.testing.assert_allclose(vals, CPP_FULL_LTMZ, rtol=1e-3)

    def test_python_fast_mass_matches_production_to_machine_precision(self):
        # Python fast_mass is a direct re-expression of the same
        # algorithm NumCountsSel.so implements (SelGLCore + SelZRule) --
        # measured 6.2e-15 agreement.
        prod = self.source.array("numcountssel", "vals")
        np.testing.assert_allclose(self._fast_mass_python(), prod, rtol=1e-9)


# ---------------------------------------------------------------------------
# Buzzard configuration: true-z photo-z bins (sigma_z = 0.001) with the
# simulation-box seam excised. The S_ij z-integrand has near-jumps at
# every zob edge and true jumps at the seam -- the case a single
# Gauss-Legendre z panel gets wrong by percent (the reason this class
# exists; the contiguous sigma_z = 0.01 wall above never exercised it).
# ---------------------------------------------------------------------------

BUZZ_DUMP_DIR = REPO / "cosmosis-models" / "real_pipeline_extract_buzzcfg_output"
HAS_BUZZ_DUMP = (BUZZ_DUMP_DIR / "sel_function" / "s_rich_stack.txt").is_file()
_BUZZ_SKIP_MSG = (f"requires the Buzzard-configuration dump at "
                  f"{BUZZ_DUMP_DIR} -- run `cosmosis "
                  "cosmosis-models/real_pipeline_extract_buzzcfg.ini` first")

BUZZ_BINS = dict(
    lam_min=np.array([20., 30., 45., 60.] * 3),
    lam_max=np.array([30., 45., 60., 200.] * 3),
    zob_min=np.array([0.20] * 4 + [0.37] * 4 + [0.50] * 4),
    zob_max=np.array([0.33] * 4 + [0.50] * 4 + [0.65] * 4),
    sigma_z=np.full(12, 0.001),
)
BUZZ_EXCL = (0.33, 0.37)

# NumCounts3d (Cuhre) on the Buzzard configuration at the same fiducial
# point as CPP_FULL_LTMZ (provenance as above). NumCounts3d has no seam
# knob, so the seam is excised through the integration volume: the sum
# of two runs, zt in [0.05, 0.33] and [0.37, 0.80]. eps_rel 1e-5 (status
# 0, err/val <= 1e-5, nregions <= 3.9e5) is the reference; eps_rel 1e-4
# is pinned too, to show Cuhre has converged on the steps.
CPP_FULL_LTMZ_BUZZ = np.array([
    952.498597226632, 378.7406179838717,
    108.69271079639861, 73.14229701448649,
    1681.8631739069683, 607.1356563852657,
    156.13114473844016, 88.83914728993355,
    2089.9281167733106, 701.8668210515436,
    165.94090365402354, 83.72360017586333,
])
CPP_FULL_LTMZ_BUZZ_EPS1E4 = np.array([
    952.4953981589554, 378.73919762006426,
    108.69227342555494, 73.14204078976518,
    1681.8621100100054, 607.1351329982775,
    156.13105593022854, 88.83908417682603,
    2089.9372309081205, 701.8709352876213,
    165.94236955353549, 83.7239503262809,
])
# NumCounts3dGpu (PAGANI), same two-volume sum.
CUDA_FULL_LTMZ_BUZZ = np.array([
    952.4869436021775, 378.7420356470596,
    108.69342256813198, 73.14311621453567,
    1682.0186077415556, 607.138801108527,
    156.16288692130016, 88.8424782073844,
    2089.9151760531604, 701.8585674091735,
    165.94350086912598, 83.72717387296525,
])


@unittest.skipUnless(HAS_BUZZ_DUMP, _BUZZ_SKIP_MSG)
class TestNumberCountsBuzzardSteps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = dm.DumpSource(str(BUZZ_DUMP_DIR))
        cls.mor = sel_kernels.mor_from_source(cls.source)
        cls.plob = sel_kernels.plob_splines_default()
        cls.hmf = dm.HMF(cls.source)
        cls.dv = dm.DVDoDz(cls.source)

    def _fast_mass_python(self, **kw):
        return dm.MassZWeights(
            self.source, n_lnm=96, n_z=kw.pop("n_z", 64), zt_lo=ZT_LOW,
            zt_hi=ZT_HIGH, lnm_lo=LNM_LOW, lnm_hi=LNM_HIGH,
            include_sci=False, **kw).norm()

    def test_cuhre_has_converged_on_the_steps(self):
        # eps_rel 1e-4 vs 1e-5 -- measured 8.8e-6.
        np.testing.assert_allclose(CPP_FULL_LTMZ_BUZZ_EPS1E4,
                                   CPP_FULL_LTMZ_BUZZ, rtol=3e-5)

    def test_cuda_full_ltmz_matches_cpp_full_ltmz(self):
        # PAGANI vs Cuhre on the step integrand -- measured 2.0e-4.
        np.testing.assert_allclose(CUDA_FULL_LTMZ_BUZZ, CPP_FULL_LTMZ_BUZZ,
                                   rtol=5e-4)

    def test_python_full_ltmz_matches_cpp_full_ltmz(self):
        # Explicit (lambda_tr, lnM, z) GL with the composite z panels and
        # the seam excised -- measured 3.0e-4.
        _, lnm_w, W = explicit_grid_core.explicit_mass_weights(
            BUZZ_BINS, self.mor, self.plob, self.hmf, self.dv,
            zt_low=ZT_LOW, zt_high=ZT_HIGH, lnm_low=LNM_LOW,
            lnm_high=LNM_HIGH, zt_excl=BUZZ_EXCL)
        np.testing.assert_allclose(W @ lnm_w, CPP_FULL_LTMZ_BUZZ, rtol=7e-4)

    def test_python_fast_mass_matches_cpp_full_ltmz(self):
        # Composite true-z rule (default) -- measured 4.0e-4.
        np.testing.assert_allclose(self._fast_mass_python(),
                                   CPP_FULL_LTMZ_BUZZ, rtol=1e-3)

    def test_production_fast_mass_matches_cpp_full_ltmz(self):
        # Production NumCountsSel.so (SelGLCore + SelZRule) -- measured 4.0e-4.
        vals = self.source.array("numcountssel", "vals")
        np.testing.assert_allclose(vals, CPP_FULL_LTMZ_BUZZ, rtol=1e-3)

    def test_python_fast_mass_matches_production_to_machine_precision(self):
        # measured 5.3e-15.
        prod = self.source.array("numcountssel", "vals")
        np.testing.assert_allclose(self._fast_mass_python(), prod, rtol=1e-9)

    def test_single_gl_panel_misses_the_steps(self):
        # Characterisation of the defect the composite rule fixes: one
        # 64-node GL panel on the bilinear S_stack table (z_rule = global,
        # the pre-2026-11 production rule) is off by +2.5% / -2.7% / +2.0%
        # in z0 / z1 / z2 (max 2.7e-2), and brute force does not converge:
        # 256 nodes are still 3.2e-2 off, because the step is
        # point-sampled on sel_function's z table and interpolated as a
        # one-cell ramp.
        for n_z in (64, 256):
            err = np.abs(self._fast_mass_python(z_rule="global", n_z=n_z)
                         / CPP_FULL_LTMZ_BUZZ - 1.0).max()
            self.assertGreater(err, 1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
