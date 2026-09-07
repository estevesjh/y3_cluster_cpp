#!/usr/bin/env python3
"""Unit tests for the CosmoSIS module contract of
``src/pipelines/des_y3/shear_1h2h/python/0d/shear1h_explicit_gl.py``.

Exercises ``setup(options)``/``execute(block, cfg)``: the per-bin array
contract, the GL-knob defaults, the REQUIRED miscentering keys, and the
documented ``NotImplementedError`` fail-loud branch for
``one_halo_physical_density`` (CLAUDE.md: "Not implemented (fails
loudly): ... the Python explicit/max mirrors").
``test/des_y3_pipeline.test.py`` already pins the pure ``compute_shear``
composition against production.
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
import shear1h_explicit_gl as mod                       # noqa: E402
from _dump_datablock import datablock_from_dump, make_options  # noqa: E402

DUMP_DIR = (Path("/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev")
           / "cosmosis-models" / "real_pipeline_extract_output")
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
        np.testing.assert_allclose(cfg["lob_centers"], dm.DEFAULT_LOB_CENTERS)
        self.assertEqual(cfg["n_lnm"], 96)
        self.assertEqual(cfg["n_z"], 64)
        self.assertEqual(cfg["n_q"], 32)
        self.assertEqual(cfg["l_lam"], 6.0)

    def test_gl_knob_overrides(self):
        entries = dict(_base_options(), n_lnm=32, n_z=16, n_q=8, l_lam=4.0,
                       lob_centers=np.array([40., 80.]))
        cfg = mod.setup(make_options(entries))
        self.assertEqual(cfg["n_lnm"], 32)
        self.assertEqual(cfg["n_z"], 16)
        self.assertEqual(cfg["n_q"], 8)
        self.assertEqual(cfg["l_lam"], 4.0)
        np.testing.assert_allclose(cfg["lob_centers"], [40., 80.])

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
        cls.block = datablock_from_dump(
            DUMP_DIR, dm,
            extra={("miscentering", "f_mis"): dm.F_MIS_DEFAULT,
                  ("miscentering", "tau_mis"): dm.TAU_MIS_DEFAULT})

    def test_execute_matches_production_within_tabulation_error(self):
        cfg = mod.setup(make_options(_base_options()))
        rc = mod.execute(self.block, cfg)
        self.assertEqual(rc, 0)
        vals = self.block[mod.OUTPUT_SECTION, "vals"]
        prod = self.block["shear1hmissel", "vals"]
        self.assertEqual(vals.shape, prod.shape)
        # README.md: measured 8.4e-4 (S_ij tabulation error), bound 5e-3.
        np.testing.assert_allclose(vals, prod, rtol=5e-3, atol=0.0)

    def test_physical_density_flag_raises_not_implemented(self):
        self.block["halomodel", "one_halo_physical_density"] = 1.0
        try:
            cfg = mod.setup(make_options(_base_options()))
            with self.assertRaises(NotImplementedError):
                mod.execute(self.block, cfg)
        finally:
            self.block["halomodel", "one_halo_physical_density"] = 0.0

    def test_missing_miscentering_section_fails_loudly(self):
        bare = datablock_from_dump(DUMP_DIR, dm)
        cfg = mod.setup(make_options(_base_options()))
        with self.assertRaises(Exception):
            mod.execute(bare, cfg)

    def test_cleanup_is_a_no_op(self):
        self.assertEqual(mod.cleanup({}), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
