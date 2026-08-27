#!/usr/bin/env python3
"""Unit tests for the CosmoSIS module contract of
``src/pipelines/des_y3/shear_1h2h/python/0d/shear1h2h_max.py``.

Uses ``real_pipeline_extract_max2h_output`` (``compute_lensing_2h = T``,
so ``haloModel/dSigma_hh`` and ``miscentering`` are actually published --
unlike ``real_pipeline_extract_output``). ``test/des_y3_pipeline.test.py``
already pins the 2h -> 0 max-model limit and the pure ``compute_shear_max``
composition; this file exercises ``setup(options)``/``execute(block,
cfg)`` themselves: the ``include_miscentering`` default/override, the
required bin_index/r_perp options, and the fail-loud
``one_halo_physical_density`` branch.
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
from shared import lensing_profiles as lp              # noqa: E402
import shear1h2h_max as mod                              # noqa: E402
from _dump_datablock import datablock_from_dump, make_options  # noqa: E402

DUMP_DIR = (Path("/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev")
           / "cosmosis-models" / "real_pipeline_extract_max2h_output")
HAS_DUMP = DUMP_DIR.is_dir()
_SKIP_MSG = f"requires the real-pipeline dump at {DUMP_DIR}"

ENVELOPE = dict(zt_low=0.05, zt_high=0.80, lnm_low=29.9336, lnm_high=36.7300)
R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
BIN_INDEX = np.arange(12)


def _base_options():
    return dict(ENVELOPE, bin_index=BIN_INDEX, r_perp=R_PERP)


class TestSetupOptionContract(unittest.TestCase):
    def test_include_miscentering_defaults_true(self):
        cfg = mod.setup(make_options(_base_options()))
        self.assertTrue(cfg["include_miscentering"])

    def test_include_miscentering_false_is_honoured(self):
        entries = dict(_base_options(), include_miscentering=False)
        cfg = mod.setup(make_options(entries))
        self.assertFalse(cfg["include_miscentering"])

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

    def test_execute_matches_production_shear1h2h_max(self):
        cfg = mod.setup(make_options(_base_options()))
        rc = mod.execute(self.block, cfg)
        self.assertEqual(rc, 0)
        vals = self.block[mod.OUTPUT_SECTION, "vals"]
        prod = self.block["shear1h2h_max", "vals"]
        self.assertEqual(vals.shape, prod.shape)
        np.testing.assert_allclose(vals, prod, rtol=1e-6, atol=0.0)

    def test_include_miscentering_false_changes_the_answer(self):
        cfg_on = mod.setup(make_options(_base_options()))
        mod.execute(self.block, cfg_on)
        vals_on = np.array(self.block[mod.OUTPUT_SECTION, "vals"])

        cfg_off = mod.setup(make_options(
            dict(_base_options(), include_miscentering=False)))
        mod.execute(self.block, cfg_off)
        vals_off = np.array(self.block[mod.OUTPUT_SECTION, "vals"])

        self.assertTrue(np.all(np.isfinite(vals_off)))
        self.assertFalse(np.allclose(vals_on, vals_off),
                         "include_miscentering=False had no effect")

    def test_physical_density_flag_raises_not_implemented(self):
        self.block["halomodel", "one_halo_physical_density"] = 1.0
        try:
            cfg = mod.setup(make_options(_base_options()))
            with self.assertRaises(NotImplementedError):
                mod.execute(self.block, cfg)
        finally:
            self.block["halomodel", "one_halo_physical_density"] = 0.0

    def test_z_resolved_weights_agree_with_the_z_contracted_ones(self):
        # z_resolved_weights' docstring promise, at module scope (not
        # exercised anywhere else): summing w2d over the z GL nodes with
        # the lnM weights left out must reproduce MassZWeights.W exactly.
        source = dm.DataBlockSource(self.block)
        lnm_x, lnm_w, z_x, w2d = mod.z_resolved_weights(
            source, n_lnm=48, n_z=32, zt_lo=ENVELOPE["zt_low"],
            zt_hi=ENVELOPE["zt_high"], lnm_lo=ENVELOPE["lnm_low"],
            lnm_hi=ENVELOPE["lnm_high"])
        weights = dm.MassZWeights(
            source, n_lnm=48, n_z=32, zt_lo=ENVELOPE["zt_low"],
            zt_hi=ENVELOPE["zt_high"], lnm_lo=ENVELOPE["lnm_low"],
            lnm_hi=ENVELOPE["lnm_high"], include_sci=True)
        np.testing.assert_allclose(w2d.sum(axis=-1), weights.W,
                                   rtol=1e-10, atol=0.0)
        np.testing.assert_array_equal(lnm_x, weights.lnm_x)
        np.testing.assert_array_equal(lnm_w, weights.lnm_w)

    def test_cleanup_is_a_no_op(self):
        self.assertEqual(mod.cleanup({}), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
