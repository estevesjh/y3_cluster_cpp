#!/usr/bin/env python3
"""Unit tests for the CosmoSIS module contract of
``src/pipelines/des_y3/shear_1h2h/python/0d/shear1h_gl.py``.

``test/des_y3_pipeline.test.py`` already pins ``compute_shear`` (the pure
mass-sum) against production ``Shear1hMisSel.so``; this file exercises
``setup(options)``/``execute(block, cfg)`` themselves: the required
``bin_index``/``r_perp``/GL envelope options, the ``lob_centers`` default,
the REQUIRED (no in-code fallback) ``miscentering/{f_mis,tau_mis})``
DataBlock keys, and the ``one_halo_physical_density`` branch
(CLAUDE.md: implemented for this backend, unlike the explicit/max
mirrors).
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
import shear1h_gl as mod                                # noqa: E402
from _dump_datablock import datablock_from_dump, make_options  # noqa: E402

DUMP_DIR = (Path("/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev")
           / "cosmosis-models" / "real_pipeline_extract_output")
HAS_DUMP = DUMP_DIR.is_dir()
_SKIP_MSG = f"requires the real-pipeline dump at {DUMP_DIR}"

ENVELOPE = dict(zt_low=0.05, zt_high=0.80, lnm_low=29.9336, lnm_high=36.7300)
R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
BIN_INDEX = np.arange(12)


def _base_options():
    return dict(ENVELOPE, bin_index=BIN_INDEX, r_perp=R_PERP)


class TestSetupOptionContract(unittest.TestCase):
    def test_defaults_and_required_fields(self):
        cfg = mod.setup(make_options(_base_options()))
        np.testing.assert_array_equal(cfg["bin_index"], BIN_INDEX)
        np.testing.assert_allclose(cfg["r_perp"], R_PERP)
        np.testing.assert_allclose(cfg["lob_centers"], dm.DEFAULT_LOB_CENTERS)
        self.assertEqual(cfg["n_lnm"], 96)
        self.assertEqual(cfg["n_z"], 64)

    def test_explicit_lob_centers_and_gl_knobs_override_defaults(self):
        entries = dict(_base_options(), lob_centers=np.array([30., 50.]),
                       n_lnm=32, n_z=16)
        cfg = mod.setup(make_options(entries))
        np.testing.assert_allclose(cfg["lob_centers"], [30., 50.])
        self.assertEqual(cfg["n_lnm"], 32)
        self.assertEqual(cfg["n_z"], 16)

    def test_missing_bin_index_or_r_perp_fails_loudly(self):
        for missing in ("bin_index", "r_perp"):
            entries = {k: v for k, v in _base_options().items()
                      if k != missing}
            with self.assertRaises(Exception, msg=f"missing {missing}"):
                mod.setup(make_options(entries))

    def test_missing_envelope_key_fails_loudly(self):
        entries = {k: v for k, v in _base_options().items() if k != "lnm_low"}
        with self.assertRaises(Exception):
            mod.setup(make_options(entries))


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestExecuteAgainstProduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = datablock_from_dump(
            DUMP_DIR, dm,
            # real_pipeline_extract_output has no `miscentering` section
            # (des_y3_pipeline.test.py's TestShear1hGl works around the
            # same gap by constructing MisMixtureProfile with the in-code
            # F_MIS_DEFAULT/TAU_MIS_DEFAULT constants directly); inject
            # the identical fiducial defaults so the REAL execute() path
            # -- which reads them from the block with no fallback -- can
            # run end-to-end against this dump.
            extra={("miscentering", "f_mis"): dm.F_MIS_DEFAULT,
                  ("miscentering", "tau_mis"): dm.TAU_MIS_DEFAULT})

    def test_execute_matches_production_shear1hmissel(self):
        cfg = mod.setup(make_options(_base_options()))
        rc = mod.execute(self.block, cfg)
        self.assertEqual(rc, 0)
        vals = self.block[mod.OUTPUT_SECTION, "vals"]
        prod = self.block["shear1hmissel", "vals"]
        self.assertEqual(vals.shape, prod.shape)
        # README.md: measured 3.1e-15 vs Shear1hMisSel.so (method=exact).
        np.testing.assert_allclose(vals, prod, rtol=1e-6, atol=0.0)

    def test_missing_miscentering_section_fails_loudly(self):
        bare = datablock_from_dump(DUMP_DIR, dm)  # no injected miscentering
        self.assertFalse(bare.has_value("miscentering", "f_mis"))
        cfg = mod.setup(make_options(_base_options()))
        with self.assertRaises(Exception):
            mod.execute(bare, cfg)

    def test_physical_density_flag_selects_the_1pz_weighted_branch(self):
        # CLAUDE.md: shear1h_gl DOES implement one_halo_physical_density
        # (unlike the explicit/max Python mirrors, which raise
        # NotImplementedError -- see shear1h_explicit_gl.test.py). Toggling
        # it must change the answer (the (1+z)^2 weight is genuinely
        # applied) while staying finite and positive.
        cfg = mod.setup(make_options(_base_options()))
        mod.execute(self.block, cfg)
        off_vals = np.array(self.block[mod.OUTPUT_SECTION, "vals"])

        self.block["halomodel", "one_halo_physical_density"] = 1.0
        try:
            mod.execute(self.block, cfg)
            on_vals = np.array(self.block[mod.OUTPUT_SECTION, "vals"])
        finally:
            self.block["halomodel", "one_halo_physical_density"] = 0.0

        self.assertTrue(np.all(np.isfinite(on_vals)))
        self.assertTrue(np.all(on_vals > 0.0))
        self.assertFalse(np.allclose(on_vals, off_vals),
                         "physical-density toggle had no effect")

    def test_cleanup_is_a_no_op(self):
        self.assertEqual(mod.cleanup({}), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
