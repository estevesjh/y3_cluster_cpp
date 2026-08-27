#!/usr/bin/env python3
"""Unit tests for the CosmoSIS module contract of
``src/pipelines/des_y3/number_counts/python/0d/numcounts_sij_gl.py``.

``test/des_y3_pipeline.test.py`` and the module's own
``validate_fast_vs_production.py`` already pin the underlying
``shared.datablock_models.MassZWeights`` physics and its agreement with
``NumCountsSel.so``; this file instead exercises the CosmoSIS
``setup(options)``/``execute(block, cfg)`` entry points themselves --
option parsing/defaults/required-key failures, and the DataBlock
read/write contract -- which had zero coverage before this change (the
whole module was 0% covered; see the coverage note in the module's own
docstring's sibling PR).

``setup()``/``execute()`` are run against a REAL ``cosmosis.datablock.
DataBlock``, either built inline (for the options contract, which does
not need any per-sample data) or replayed from the pinned
``real_pipeline_extract_output`` dump via ``test/_dump_datablock.py``
(for the execute()/physics leg) -- not the offline ``DumpSource``, which
the module's own ``execute()`` cannot accept directly (see
``_dump_datablock.py``'s docstring).
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
import numcounts_sij_gl as mod                          # noqa: E402
from _dump_datablock import datablock_from_dump, make_options  # noqa: E402

# Absolute path into the main checkout: fixture dumps are gitignored and
# not present in this worktree (see CLAUDE.md's "Fixture data" note).
DUMP_DIR = (Path("/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev")
           / "cosmosis-models" / "real_pipeline_extract_output")
HAS_DUMP = DUMP_DIR.is_dir()
_SKIP_MSG = f"requires the real-pipeline dump at {DUMP_DIR}"

REQUIRED_ENVELOPE = dict(zt_low=0.05, zt_high=0.80,
                         lnm_low=29.9336, lnm_high=36.7300)


class TestSetupOptionContract(unittest.TestCase):
    """``setup(options)``: required GL envelope, bin_index/n_lnm/n_z defaults."""

    def test_minimal_options_produce_the_documented_defaults(self):
        cfg = mod.setup(make_options(REQUIRED_ENVELOPE))
        self.assertIsNone(cfg["bin_index"])
        self.assertEqual(cfg["n_lnm"], 96)
        self.assertEqual(cfg["n_z"], 64)
        for key, val in REQUIRED_ENVELOPE.items():
            self.assertEqual(cfg[key], val)

    def test_bin_index_is_read_as_an_int_array(self):
        entries = dict(REQUIRED_ENVELOPE, bin_index=np.array([0, 3, 7]))
        cfg = mod.setup(make_options(entries))
        np.testing.assert_array_equal(cfg["bin_index"], [0, 3, 7])
        self.assertEqual(cfg["bin_index"].dtype, np.int64)

    def test_n_lnm_and_n_z_overrides_are_honoured(self):
        entries = dict(REQUIRED_ENVELOPE, n_lnm=48, n_z=32)
        cfg = mod.setup(make_options(entries))
        self.assertEqual(cfg["n_lnm"], 48)
        self.assertEqual(cfg["n_z"], 32)

    def test_missing_required_envelope_key_fails_loudly(self):
        for missing in ("zt_low", "zt_high", "lnm_low", "lnm_high"):
            entries = {k: v for k, v in REQUIRED_ENVELOPE.items()
                      if k != missing}
            with self.assertRaises(Exception, msg=f"missing {missing}"):
                mod.setup(make_options(entries))


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestExecuteAgainstProduction(unittest.TestCase):
    """``execute(block, cfg)`` end-to-end against a real dump."""

    @classmethod
    def setUpClass(cls):
        cls.block = datablock_from_dump(DUMP_DIR, dm)

    def test_execute_matches_production_for_every_wall_bin(self):
        cfg = mod.setup(make_options(REQUIRED_ENVELOPE))
        rc = mod.execute(self.block, cfg)
        self.assertEqual(rc, 0)
        vals = self.block[mod.OUTPUT_SECTION, "vals"]
        prod = self.block["numcountssel", "vals"]
        self.assertEqual(vals.shape, prod.shape)
        # Same algorithm/GL nodes/S_stack interpolation as NumCountsSel.so
        # (README.md: measured 2.4e-15); a loose 1e-6 bound keeps this
        # robust to the dump's own stored float precision.
        np.testing.assert_allclose(vals, prod, rtol=1e-6, atol=0.0)

    def test_execute_with_an_explicit_bin_subset_matches_the_full_norm(self):
        subset = np.array([2, 5, 9])
        cfg = mod.setup(make_options(dict(REQUIRED_ENVELOPE,
                                          bin_index=subset)))
        mod.execute(self.block, cfg)
        vals = self.block[mod.OUTPUT_SECTION, "vals"]
        self.assertEqual(vals.size, subset.size)

        full_cfg = mod.setup(make_options(REQUIRED_ENVELOPE))
        mod.execute(self.block, full_cfg)
        full_vals = self.block[mod.OUTPUT_SECTION, "vals"]
        np.testing.assert_allclose(vals, full_vals[subset],
                                   rtol=0.0, atol=0.0)

    def test_cleanup_is_a_no_op(self):
        self.assertEqual(mod.cleanup({}), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
