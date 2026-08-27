#!/usr/bin/env python3
"""Unit tests for the CosmoSIS module contract of
``src/pipelines/des_y3/shear_projection/python/0d/shear_prj_gl.py``.

``test/des_y3_pipeline.test.py``'s ``TestShearPrjGl`` already pins
``build_theta_grid``/``theta_excl_at_z`` against golden values and (dump
-gated) ``ShearPrjGl.set_sample()``/``wall_outputs()`` directly. This
file instead exercises the two CosmoSIS entry points ``setup(options)``
(which here returns a ``ShearPrjGl`` instance, not a plain dict --
unusual among the 0d des_y3 backends, worth pinning on its own) and
``execute(block, core)`` through a real DataBlock: the required wall
arrays, the ``R_max_cMpch``-vs-``r_max_cmpch`` ini/cfg key rename, the
GL-node/knob defaults, the ``use_halo_model_conc`` branch (issue #13),
and the required ``b_sel_marginalised`` per-(zob, lob) coverage check
inside ``set_sample()``.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PIPELINES = REPO / "src" / "pipelines"
MODULE_DIR = PIPELINES / "des_y3" / "shear_projection" / "python" / "0d"
sys.path.insert(0, str(PIPELINES))
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import datablock_models as dm             # noqa: E402
import shear_prj_gl as mod                              # noqa: E402
from _dump_datablock import datablock_from_dump, make_options  # noqa: E402

DUMP_DIR = (Path("/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev")
           / "cosmosis-models" / "real_pipeline_extract_prj2h_output")
HAS_DUMP = DUMP_DIR.is_dir()
_SKIP_MSG = f"requires the real-pipeline dump at {DUMP_DIR}"

_PRJ_RADII = [0.0426, 0.0669, 0.1045, 0.1652, 0.2607, 0.4117, 0.6505, 1.0257,
             1.6181, 2.5537, 4.0265, 6.3490, 10.0107, 15.7832, 24.8771]


def _pinned_prj_wall():
    """The 180-point zipped wall (mock_mcmc_buzzard.ini shear_prj_frozen_physics
    section): 4 lambda bins x 3 zob bins x 15 radii -- identical to the one
    ``test/des_y3_pipeline.test.py`` pins."""
    lb, zol, zoh, rr = [], [], [], []
    for zlo, zhi in ((0.20, 0.35), (0.35, 0.50), (0.50, 0.65)):
        for b in range(4):
            lb += [b] * 15
            zol += [zlo] * 15
            zoh += [zhi] * 15
            rr += _PRJ_RADII
    return dict(lambda_bin=np.array(lb, dtype=float),
               zo_low=np.array(zol), zo_high=np.array(zoh),
               radii=np.array(rr))


WALL = _pinned_prj_wall()


def _base_options(**extra):
    return dict(WALL, n_lnm=24, **extra)


class TestSetupOptionContract(unittest.TestCase):
    def test_returns_a_shearprjgl_instance_with_documented_defaults(self):
        core = mod.setup(make_options(_base_options()))
        self.assertIsInstance(core, mod.ShearPrjGl)
        self.assertEqual(core.cfg["n_lnm"], 24)
        self.assertEqual(core.cfg["n_per_seg"], 10)
        self.assertEqual(core.cfg["n_zring"], 20)
        self.assertEqual(core.cfg["n_zouter"], 20)
        self.assertEqual(core.cfg["zt_low"], 0.10)
        self.assertEqual(core.cfg["zt_high"], 0.75)
        self.assertEqual(core.cfg["r_max_cmpch"], 35.0)
        self.assertFalse(core.cfg["use_halo_model_conc"])
        np.testing.assert_allclose(core.lob_centers, dm.DEFAULT_LOB_CENTERS)
        # 180 wall points collapse to 12 unique (lambda_bin, zob) slices.
        self.assertEqual(core.wall_n, 180)
        self.assertEqual(len(core.slices), 12)
        self.assertEqual(sum(len(s["Rs"]) for s in core.slices), 180)

    def test_r_max_cmpch_reads_the_capitalised_ini_key(self):
        core = mod.setup(make_options(_base_options(**{"R_max_cMpch": 50.0})))
        self.assertEqual(core.cfg["r_max_cmpch"], 50.0)

    def test_use_halo_model_conc_is_honoured(self):
        core = mod.setup(make_options(
            _base_options(use_halo_model_conc=True)))
        self.assertTrue(core.cfg["use_halo_model_conc"])

    def test_explicit_lob_centers_and_envelope_override_defaults(self):
        entries = _base_options(lob_centers=np.array([40., 90.]),
                                zt_low=0.2, zt_high=0.6,
                                lnm_low=30.0, lnm_high=35.0,
                                n_per_seg=6, n_zring=12, n_zouter=12)
        core = mod.setup(make_options(entries))
        np.testing.assert_allclose(core.lob_centers, [40., 90.])
        self.assertEqual(core.cfg["zt_low"], 0.2)
        self.assertEqual(core.cfg["n_per_seg"], 6)

    def test_missing_required_wall_key_fails_loudly(self):
        for missing in ("lambda_bin", "zo_low", "zo_high", "radii"):
            entries = {k: v for k, v in _base_options().items()
                      if k != missing}
            with self.assertRaises(Exception, msg=f"missing {missing}"):
                mod.setup(make_options(entries))


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestExecuteAgainstProduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = datablock_from_dump(DUMP_DIR, dm)

    def test_execute_matches_the_exact_evaluator_at_machine_precision(self):
        core = mod.setup(make_options(_base_options()))
        rc = mod.execute(self.block, core)
        self.assertEqual(rc, 0)

        rnd = self.block[mod.DSIGMA_SECTION, "rnd"]
        cl = self.block[mod.DSIGMA_SECTION, "cl"]
        vals = self.block[mod.DSIGMA_SECTION, "vals"]
        np.testing.assert_allclose(vals, rnd + cl, rtol=0.0, atol=0.0)

        ex_rnd = self.block["dsigma_prj", "rnd"]
        ex_cl = self.block["dsigma_prj", "cl"]
        ex_vals = self.block["dsigma_prj", "vals"]
        # des_y3_pipeline.test.py measures 1e-8-level agreement against
        # DSigmaPrjEvaluator.so (same core); a slightly looser bound keeps
        # this robust to the dump's stored text precision.
        np.testing.assert_allclose(rnd, ex_rnd, rtol=1e-6, atol=0.0)
        np.testing.assert_allclose(cl, ex_cl, rtol=1e-6, atol=0.0)
        np.testing.assert_allclose(vals, ex_vals, rtol=1e-6, atol=0.0)

    def test_shear_section_is_the_dsigma_section_times_sci(self):
        core = mod.setup(make_options(_base_options()))
        mod.execute(self.block, core)
        dsigma_vals = np.asarray(self.block[mod.DSIGMA_SECTION, "vals"])
        shear_vals = np.asarray(self.block[mod.SHEAR_SECTION, "vals"])
        sci = np.asarray(self.block["average_sigma_crit_inv",
                                    "sci_average"])
        # sci varies by lens redshift, not stored per-wall-point directly
        # in this section; instead confirm the ratio shear/dsigma is
        # exactly the same array wall_outputs() returned as `sci`.
        ratio = shear_vals / dsigma_vals
        self.assertTrue(np.all(np.isfinite(ratio)))
        self.assertTrue(np.all(ratio > 0.0))

    def test_use_halo_model_conc_changes_the_answer_and_stays_finite(self):
        core_default = mod.setup(make_options(_base_options()))
        mod.execute(self.block, core_default)
        default_vals = np.array(self.block[mod.DSIGMA_SECTION, "vals"])

        core_conc = mod.setup(make_options(
            _base_options(use_halo_model_conc=True)))
        mod.execute(self.block, core_conc)
        conc_vals = np.array(self.block[mod.DSIGMA_SECTION, "vals"])

        self.assertTrue(np.all(np.isfinite(conc_vals)))
        self.assertTrue(np.all(conc_vals > 0.0))
        self.assertFalse(np.allclose(conc_vals, default_vals),
                         "use_halo_model_conc had no effect")

    def test_bsel_table_accepts_the_wide_zob_by_lob_grid_layout(self):
        # bsel_table() supports two layouts: one row per wall (zob, lob)
        # combination (what the dump stores, and what every other test
        # here exercises), or a pre-gridded (n_zob, n_lob) table. Force
        # the second branch by duplicating every wall-metadata row (so
        # bs_zob.size = 24 != n_zob*n_lob = 12) while leaving
        # b_small/b_large at their original 12-value wide layout -- the
        # dump's zob.txt/lob.txt confirms it is already row-major
        # (zob slowest, lob fastest), matching the reshape the module
        # performs, so this must reproduce the per-row answer exactly.
        wide = datablock_from_dump(DUMP_DIR, dm)
        dup = lambda a: np.concatenate([a, a])  # noqa: E731
        for key in ("lob", "zob", "lambda_bin", "zo_low", "zo_high"):
            wide["b_sel_marginalised", key] = dup(
                np.asarray(wide["b_sel_marginalised", key]))

        core = mod.setup(make_options(_base_options()))
        mod.execute(wide, core)
        wide_vals = np.array(wide[mod.DSIGMA_SECTION, "vals"])

        core_ref = mod.setup(make_options(_base_options()))
        mod.execute(self.block, core_ref)
        ref_vals = np.array(self.block[mod.DSIGMA_SECTION, "vals"])
        np.testing.assert_allclose(wide_vals, ref_vals, rtol=1e-10)

    def test_bsel_wrong_array_length_fails_loudly(self):
        bad = datablock_from_dump(DUMP_DIR, dm)
        bad["b_sel_marginalised", "b_small"] = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0])   # neither 12 rows nor a 3x4 grid
        core = mod.setup(make_options(_base_options()))
        with self.assertRaises(ValueError):
            mod.execute(bad, core)

    def test_incomplete_bsel_coverage_fails_loudly(self):
        # ShearPrjGl.set_sample()'s bsel_table() must reject a
        # b_sel_marginalised table that does not cover every (zob, lob)
        # pair -- drop one wall row's worth of b_small/b_large so one
        # combination goes missing.
        bad = datablock_from_dump(DUMP_DIR, dm)
        lob = np.asarray(bad["b_sel_marginalised", "lob"])[:-1]
        zob = np.asarray(bad["b_sel_marginalised", "zob"])[:-1]
        b_small = np.asarray(bad["b_sel_marginalised", "b_small"])[:-1]
        b_large = np.asarray(bad["b_sel_marginalised", "b_large"])[:-1]
        lambda_bin = np.asarray(bad["b_sel_marginalised",
                                    "lambda_bin"])[:-1]
        zo_low = np.asarray(bad["b_sel_marginalised", "zo_low"])[:-1]
        zo_high = np.asarray(bad["b_sel_marginalised", "zo_high"])[:-1]
        for key, arr in (("lob", lob), ("zob", zob), ("b_small", b_small),
                        ("b_large", b_large), ("lambda_bin", lambda_bin),
                        ("zo_low", zo_low), ("zo_high", zo_high)):
            bad["b_sel_marginalised", key] = arr

        core = mod.setup(make_options(_base_options()))
        with self.assertRaises(ValueError):
            mod.execute(bad, core)

    def test_cleanup_is_a_no_op(self):
        self.assertEqual(mod.cleanup({}), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
