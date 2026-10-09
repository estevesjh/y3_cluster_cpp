#!/usr/bin/env python3
"""Unit tests for the CosmoSIS module contract of
``src/pipelines/des_y3/shear_1h2h/python/0d/shear1h2h_max.py``.

Uses ``real_pipeline_extract_max2h_output`` (``compute_lensing_2h = T``,
so ``haloModel/dSigma_hh`` and ``miscentering`` are actually published --
unlike ``real_pipeline_extract_output``). ``test/des_y3_pipeline.test.py``
already pins the 2h -> 0 max-model limit and the pure ``compute_shear_max``
composition; this file exercises ``setup(options)``/``execute(block,
cfg)`` themselves: the ``include_miscentering`` default/override, the
required bin_index/r_perp options, and the ``one_halo_physical_density``
1-halo z-resolution branch.
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
from systematics.selection_boost.Bsel import (          # noqa: E402
    Shear1h2hMaxSelCostanzi26,
    Shear1h2hMaxSelSunayama23,
)
from systematics.selection_boost.BselCostanzi26 import BselCostanzi26  # noqa: E402
from systematics.selection_boost.BselSunayama23 import BselSunayama23  # noqa: E402

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
    def test_bsel_defaults_disabled(self):
        cfg = mod.setup(make_options(_base_options()))
        self.assertFalse(cfg["bsel"])

    def test_bsel_tag_is_read(self):
        cfg = mod.setup(make_options(dict(
            _base_options(), bsel=True, bsel_model="Costanzi26")))
        self.assertTrue(cfg["bsel"])
        self.assertEqual(cfg["bsel_model"], "Costanzi26")
        self.assertNotIn("bsel_section", cfg)

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


class TestSelectedProfileComposition(unittest.TestCase):
    class _Profile:
        def _one(self, _bin, radii, masses, q=1.0):
            return np.full((radii.shape[0], masses.shape[1]), 2.0)

        def _bias(self, masses, redshifts):
            return np.ones((masses.shape[0], redshifts.shape[1]))

        def _hh(self, radii, redshifts):
            return np.full((radii.shape[0], redshifts.shape[1]), 3.0)

    def test_each_selected_consumer_is_applied_after_the_max(self):
        profile = self._Profile()
        args = (profile, np.array([0.0]), np.array([1.0]),
                np.array([0.0]), np.ones((1, 1, 1)), np.array([0]),
                np.array([1.0]))

        costanzi = Shear1h2hMaxSelCostanzi26(
            BselCostanzi26(0.1, 0.92, -0.53, 4.1))
        got_costanzi = mod.compute_shear_max(
            *args, selected_profile=costanzi, lob_centers=np.array([40.0]))
        expected_costanzi = 3.0 * costanzi.model(1.0, 40.0, 0.0)
        np.testing.assert_allclose(got_costanzi, [expected_costanzi])

        sunayama = Shear1h2hMaxSelSunayama23(
            BselSunayama23([0.45], [3.5], -0.1, [0]))
        got_sunayama = mod.compute_shear_max(
            *args, selected_profile=sunayama, lob_centers=np.array([40.0]))
        expected_sunayama = 3.0 * sunayama.model(1.0, 0)
        np.testing.assert_allclose(got_sunayama, [expected_sunayama])


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

    def test_physical_density_flag_selects_the_1pz_weighted_branch(self):
        # one_halo_physical_density IS wired into this z-resolved mirror
        # (unlike the z-contracted-early shear1h_explicit_gl): the 2-halo
        # term already forces a z-resolved 1-halo evaluation, so no
        # restructuring is needed -- see compute_shear_max's docstring.
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
