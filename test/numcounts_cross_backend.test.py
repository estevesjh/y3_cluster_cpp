#!/usr/bin/env python3
"""Cross-backend / cross-strategy consistency for DES Y3 number counts.

Verifies that every implemented (strategy, backend) cell for the number
counts observable agrees with every other cell, per the matrix in
docs/source/pipeline_organization.md:

    full_ltmz  -> Python, C++, CUDA   (explicit-selection accuracy references)
    fast_mass  -> Python, C++ (production `NumCountsSel.so`, by identity)

The reference is the C++ `full_ltmz` backend (`NumCountsFullLtmz.so`,
adaptive Cuhre, eps_rel=1e-4) -- the project's designated accuracy
reference (docs/source/pipeline_organization.md "Validation policy").
Its values, and the CUDA `full_ltmz` backend's (`NumCountsFullLtmzGpu.so`,
PAGANI), are hard-coded below: reproducing them needs a full CosmoSIS
pipeline run of the compiled .so modules (and, for CUDA, a GPU), which
this lightweight, portable unittest does not attempt. They were
generated (2026-08-12, SLURM job 56780752 companion run) from a single
pipeline execution -- `gpu_smoke.ini`, modules `consistency prj_params
GrowthFactor cp_camb MfTinker halo_model average_sigma_crit_inv
sel_function NumCountsSel numcounts_full_ltmz NumCountsFullLtmz
NumCountsFullLtmzGpu` -- with the SAME `mock_mcmc_widePlanck_values.ini`
fiducial point and the SAME pinned 12-bin wall used everywhere else in
this repo's tests.

The Python `full_ltmz` and `fast_mass` backends, and production's own
saved output, are all computed/read LIVE from this repo's checked-in
fiducial dump (docs/figs/real_pipeline_extract_output) -- verified
(2026-08-13) to reproduce the external pipeline run's Python full_ltmz
numbers to ~1e-7 relative, confirming the checked-in dump alone is a
sufficient, portable substitute for the full external run. Only the
two backends that cannot be cheaply re-invoked from a plain Python
process (they need real .so modules through a full CosmoSIS pipeline)
are pinned as literals.

To regenerate CPP_FULL_LTMZ / CUDA_FULL_LTMZ after a real change to
either backend: run a pipeline with `NumCountsFullLtmz`/
`NumCountsFullLtmzGpu` appended to `docs/figs/real_pipeline_extract.ini`
(same bin wall, `zt_low=0.05, zt_high=0.80, lnm_low=29.9336,
lnm_high=36.7300`, `lt_low`/`lt_high` per bin =
{0.1, 4*lam_max}) and read back `numcountsfullltmz(gpu)/vals`.
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
from numcounts_full_ltmz import compute_counts  # noqa: E402

DUMP_DIR = REPO / "docs" / "figs" / "real_pipeline_extract_output"
HAS_DUMP = (DUMP_DIR / "matter_power_lin" / "p_k.txt").is_file()
_SKIP_MSG = (f"requires a real-pipeline dump at {DUMP_DIR} -- run "
            "`cosmosis docs/figs/real_pipeline_extract.ini` first")

# The pinned 12-bin wall (mock_mcmc_buzzard.ini sel_function section),
# identical to validate_against_fiducial.py / validate_vs_production.py.
BINS = dict(
    lam_min=np.array([20., 30., 45., 60.] * 3),
    lam_max=np.array([30., 45., 60., 200.] * 3),
    zob_min=np.array([0.20] * 4 + [0.35] * 4 + [0.50] * 4),
    zob_max=np.array([0.35] * 4 + [0.50] * 4 + [0.65] * 4),
    sigma_z=np.full(12, 0.03),
)
ZT_LOW, ZT_HIGH = 0.05, 0.80
LNM_LOW, LNM_HIGH = 29.9336, 36.7300

# C++ full_ltmz (NumCountsFullLtmz.so) -- THE reference. See module
# docstring for provenance.
CPP_FULL_LTMZ = np.array([
    1.380690148279805726e+03, 5.137561651299496361e+02,
    1.414299878149995209e+02, 9.168397349335343449e+01,
    2.300356590375406995e+03, 7.848939444311833995e+02,
    1.962481919281859746e+02, 1.102119951375328952e+02,
    2.475944300931746511e+03, 7.793651262731299312e+02,
    1.778104398634395693e+02, 8.783936722369612937e+01,
])

# CUDA full_ltmz (NumCountsFullLtmzGpu.so, PAGANI).
CUDA_FULL_LTMZ = np.array([
    1.380700754976703820e+03, 5.137602886997940459e+02,
    1.414300187864491818e+02, 9.167849217128190276e+01,
    2.300342731150394229e+03, 7.848978224091276843e+02,
    1.962482766981219697e+02, 1.102129609017462570e+02,
    2.475995092060699790e+03, 7.793647183037339801e+02,
    1.778124481656746525e+02, 8.784365020955073078e+01,
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
        # adaptive Cuhre) over the same physics -- measured 4.9e-4.
        vals = compute_counts(BINS, self.mor, self.plob, self.hmf, self.dv,
                              zt_low=ZT_LOW, zt_high=ZT_HIGH,
                              lnm_low=LNM_LOW, lnm_high=LNM_HIGH)
        np.testing.assert_allclose(vals, CPP_FULL_LTMZ, rtol=7e-4)

    def test_cuda_full_ltmz_matches_cpp_full_ltmz(self):
        # Two independent adaptive integrators (PAGANI vs Cuhre) over the
        # identical integrand -- measured 6.0e-5.
        np.testing.assert_allclose(CUDA_FULL_LTMZ, CPP_FULL_LTMZ, rtol=1e-4)

    def test_python_fast_mass_matches_cpp_full_ltmz(self):
        # fast_mass's S_ij tabulation vs the explicit triple integral --
        # measured 7.6e-4.
        np.testing.assert_allclose(self._fast_mass_python(), CPP_FULL_LTMZ,
                                   rtol=1.5e-3)

    def test_production_fast_mass_matches_cpp_full_ltmz(self):
        # Production NumCountsSel.so's own saved output vs full_ltmz --
        # measured 1.1e-3 (same S_ij-tabulation error as the Python
        # fast_mass check plus the Cuhre tolerance).
        vals = self.source.array("numcountssel", "vals")
        np.testing.assert_allclose(vals, CPP_FULL_LTMZ, rtol=1.5e-3)

    def test_python_fast_mass_matches_production_to_machine_precision(self):
        # Python fast_mass is a direct re-expression of the same
        # algorithm NumCountsSel.so implements (SelGLCore) -- measured
        # 2.4e-15 agreement.
        prod = self.source.array("numcountssel", "vals")
        np.testing.assert_allclose(self._fast_mass_python(), prod, rtol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
