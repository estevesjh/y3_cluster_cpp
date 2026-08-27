#!/usr/bin/env python3
"""Unit tests for the CosmoSIS module contract of
``src/pipelines/des_y3/shear_1h2h/python/0d/shear1h2h_max_explicit_gl.py``.

This module had ZERO coverage before this change (not even imported by
``des_y3_pipeline.test.py``). Uses ``real_pipeline_extract_max2h_output``
(``compute_lensing_2h = T``). Exercises ``setup(options)``/
``execute(block, cfg)``: the per-bin array contract, GL-knob defaults,
``include_miscentering``, and the fail-loud ``one_halo_physical_density``
branch -- plus a direct numeric check against the fixed-GL
``shear1h2h_max`` fast path, isolating the S_ij-tabulation error the
module's own docstring documents.
"""
from __future__ import annotations

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

from shared import datablock_models as dm             # noqa: E402
import shear1h2h_max_explicit_gl as mod                  # noqa: E402
from _dump_datablock import datablock_from_dump, make_options  # noqa: E402

DUMP_DIR = (Path("/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev")
           / "cosmosis-models" / "real_pipeline_extract_max2h_output")
HAS_DUMP = DUMP_DIR.is_dir()
_SKIP_MSG = f"requires the real-pipeline dump at {DUMP_DIR}"

BINS_12 = dict(
    lam_min=np.array([20., 30., 45., 60.] * 3),
    lam_max=np.array([30., 45., 60., 200.] * 3),
    zob_min=np.array([0.20] * 4 + [0.35] * 4 + [0.50] * 4),
    zob_max=np.array([0.35] * 4 + [0.50] * 4 + [0.65] * 4),
    sigma_z=np.full(12, 0.03),
)
ENVELOPE = dict(zt_low=0.05, zt_high=0.80, lnm_low=29.9336, lnm_high=36.7300)
R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
BIN_INDEX = np.arange(12)


def _base_options():
    return dict(BINS_12, **ENVELOPE, bin_index=BIN_INDEX, r_perp=R_PERP)


class TestSetupOptionContract(unittest.TestCase):
    def test_defaults(self):
        cfg = mod.setup(make_options(_base_options()))
        self.assertEqual(cfg["n_lnm"], 96)
        self.assertEqual(cfg["n_z"], 64)
        self.assertEqual(cfg["n_q"], 32)
        self.assertEqual(cfg["l_lam"], 6.0)
        self.assertTrue(cfg["include_miscentering"])

    def test_overrides(self):
        entries = dict(_base_options(), n_lnm=32, n_z=16, n_q=8, l_lam=5.0,
                       include_miscentering=False,
                       lob_centers=np.array([30., 60.]))
        cfg = mod.setup(make_options(entries))
        self.assertEqual(cfg["n_lnm"], 32)
        self.assertFalse(cfg["include_miscentering"])
        np.testing.assert_allclose(cfg["lob_centers"], [30., 60.])

    def test_missing_bin_index_or_r_perp_fails_loudly(self):
        for missing in ("bin_index", "r_perp"):
            entries = {k: v for k, v in _base_options().items()
                      if k != missing}
            with self.assertRaises(Exception, msg=f"missing {missing}"):
                mod.setup(make_options(entries))


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestExecuteAgainstProduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = datablock_from_dump(DUMP_DIR, dm)

    def test_execute_matches_the_fixed_gl_fast_path_within_tabulation_error(self):
        cfg = mod.setup(make_options(_base_options()))
        rc = mod.execute(self.block, cfg)
        self.assertEqual(rc, 0)
        vals = self.block[mod.OUTPUT_SECTION, "vals"]
        prod = self.block["shear1h2h_max", "vals"]
        self.assertEqual(vals.shape, prod.shape)
        # Measured 8.4e-4 (same S_ij-tabulation error as the 1-halo pair).
        np.testing.assert_allclose(vals, prod, rtol=5e-3, atol=0.0)

    def test_physical_density_flag_raises_not_implemented(self):
        self.block["halomodel", "one_halo_physical_density"] = 1.0
        try:
            cfg = mod.setup(make_options(_base_options()))
            with self.assertRaises(NotImplementedError):
                mod.execute(self.block, cfg)
        finally:
            self.block["halomodel", "one_halo_physical_density"] = 0.0

    def test_include_miscentering_false_changes_the_answer(self):
        cfg_on = mod.setup(make_options(_base_options()))
        mod.execute(self.block, cfg_on)
        vals_on = np.array(self.block[mod.OUTPUT_SECTION, "vals"])

        cfg_off = mod.setup(make_options(
            dict(_base_options(), include_miscentering=False)))
        mod.execute(self.block, cfg_off)
        vals_off = np.array(self.block[mod.OUTPUT_SECTION, "vals"])

        self.assertTrue(np.all(np.isfinite(vals_off)))
        self.assertFalse(np.allclose(vals_on, vals_off))

    def test_cleanup_is_a_no_op(self):
        self.assertEqual(mod.cleanup({}), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
