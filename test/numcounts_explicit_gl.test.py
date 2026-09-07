#!/usr/bin/env python3
"""Unit tests for the CosmoSIS module contract of
``src/pipelines/des_y3/number_counts/python/0d/numcounts_explicit_gl.py``.

``test/des_y3_pipeline.test.py`` already pins ``compute_counts`` (the
pure explicit-3d integral) against production; this file instead
exercises ``setup(options)``/``execute(block, cfg)`` themselves: the
per-bin array size-consistency check, the ``zt_low``/``zt_high``/
``lnm_low``/``lnm_high`` scalar-or-first-element fallback, the
``n_lnm``/``n_z``/``n_q``/``l_lam`` defaults, and the real DataBlock
read/write path (including the ``plob_ltr_params`` absent -> in-code
default fallback the module's own ``sel_function._make_plob_splines``
takes when a dump/pipeline never published that optional section).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PIPELINES = REPO / "src" / "pipelines"
MODULE_DIR = PIPELINES / "des_y3" / "number_counts" / "python" / "0d"
sys.path.insert(0, str(PIPELINES))
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import datablock_models as dm             # noqa: E402
import numcounts_explicit_gl as mod                     # noqa: E402
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


class TestSetupOptionContract(unittest.TestCase):
    def test_defaults_match_the_documented_fallbacks(self):
        cfg = mod.setup(make_options(BINS_12))
        self.assertEqual(cfg["zt_low"], 0.05)
        self.assertEqual(cfg["zt_high"], 0.80)
        self.assertAlmostEqual(cfg["lnm_low"], np.log(1.0e13))
        self.assertAlmostEqual(cfg["lnm_high"], np.log(9.0e15))
        self.assertEqual(cfg["n_lnm"], 96)
        self.assertEqual(cfg["n_z"], 64)
        self.assertEqual(cfg["n_q"], 32)
        self.assertEqual(cfg["l_lam"], 6.0)

    def test_explicit_envelope_and_quadrature_knobs_override_defaults(self):
        entries = dict(BINS_12, **ENVELOPE, n_lnm=48, n_z=24, n_q=16,
                       l_lam=5.0)
        cfg = mod.setup(make_options(entries))
        for key, val in ENVELOPE.items():
            self.assertEqual(cfg[key], val)
        self.assertEqual(cfg["n_lnm"], 48)
        self.assertEqual(cfg["n_z"], 24)
        self.assertEqual(cfg["n_q"], 16)
        self.assertEqual(cfg["l_lam"], 5.0)

    def test_mismatched_bin_axis_sizes_raise_value_error(self):
        bad = dict(BINS_12)
        bad["lam_max"] = BINS_12["lam_max"][:-1]   # 11 vs 12
        with self.assertRaises(ValueError):
            mod.setup(make_options(bad))

    def test_zt_low_accepts_a_bare_scalar_double_as_well_as_an_array(self):
        # sel_function._read_scalar_or_first tries the array reader first,
        # then falls back to options.get_double -- exercise both branches.
        entries = dict(BINS_12)
        entries["zt_low"] = 0.10          # scalar, not a 1-element array
        cfg = mod.setup(make_options(entries))
        self.assertEqual(cfg["zt_low"], 0.10)


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestExecuteAgainstProduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = datablock_from_dump(DUMP_DIR, dm)

    def test_execute_matches_production_within_tabulation_error(self):
        cfg = mod.setup(make_options(dict(BINS_12, **ENVELOPE)))
        rc = mod.execute(self.block, cfg)
        self.assertEqual(rc, 0)
        vals = self.block[mod.OUTPUT_SECTION, "vals"]
        prod = self.block["numcountssel", "vals"]
        self.assertEqual(vals.shape, prod.shape)
        # Validated tolerance from validate_explicit_vs_production.py:
        # measured 7.6e-4 (S_ij tabulation error), bound at 5e-3.
        np.testing.assert_allclose(vals, prod, rtol=5e-3, atol=0.0)

    def test_falls_back_to_default_plob_splines_when_section_absent(self):
        # real_pipeline_extract_output never published plob_ltr_params
        # (only a `prj_params` shim run would), so this exercises
        # sel_function._make_plob_splines' except-branch fallback to
        # PrjParams.default() through the live execute() path.
        self.assertFalse(self.block.has_value("plob_ltr_params", "z"))
        cfg = mod.setup(make_options(dict(BINS_12, **ENVELOPE)))
        mod.execute(self.block, cfg)
        vals = self.block[mod.OUTPUT_SECTION, "vals"]
        self.assertTrue(np.all(np.isfinite(vals)))
        self.assertTrue(np.all(vals > 0.0))

    def test_cleanup_is_a_no_op(self):
        self.assertEqual(mod.cleanup({}), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
