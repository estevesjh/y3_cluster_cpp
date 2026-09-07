#!/usr/bin/env python3
"""Unit tests for the CosmoSIS module contract of
``src/pipelines/des_y3/shear_1h2h/python/0d/shear1h_radial_series.py``.

``test/des_y3_pipeline.test.py`` already pins the offline table mixture
algebra (``RadialSeriesTable``, ``evaluate_series``, ``u_mix`` endpoints).
This file exercises ``setup(options)``/``execute(block, cfg)`` themselves:
the ``ell_max`` validation, the ``table`` option, the required
miscentering keys, the multi-key ``execute()`` DataBlock write contract
(``vals``/``norm``/``y_eff``/``mu2``/``mu3``), and a genuine truncation
check of the live ``execute()`` output against
``validate_radial_series.py``'s own ``exact_mass_integral`` reference
(the exact fixed-GL mass sum of the SAME fixed-c profile family) at its
already-established 0.75% tolerance -- not a fresh, invented bound.
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
import shear1h_radial_series as mod                     # noqa: E402
import nfw_profile_family as pf                          # noqa: E402
from validate_radial_series import exact_mass_integral   # noqa: E402
from _dump_datablock import datablock_from_dump, make_options  # noqa: E402

DUMP_DIR = (Path("/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev")
           / "cosmosis-models" / "real_pipeline_extract_output")
HAS_DUMP = DUMP_DIR.is_dir()
_SKIP_MSG = f"requires the real-pipeline dump at {DUMP_DIR}"

TRUNC_TOL_ELL2 = 7.5e-3   # validate_radial_series.py's own established bound

ENVELOPE = dict(zt_low=0.05, zt_high=0.80, lnm_low=29.9336, lnm_high=36.7300)
R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
BIN_INDEX = np.arange(12)


def _base_options():
    return dict(ENVELOPE, bin_index=BIN_INDEX, r_perp=R_PERP)


class TestSetupOptionContract(unittest.TestCase):
    def test_defaults(self):
        cfg = mod.setup(make_options(_base_options()))
        self.assertEqual(cfg["n_lnm"], 96)
        self.assertEqual(cfg["n_z"], 64)
        self.assertEqual(cfg["ell_max"], 2)
        self.assertIsInstance(cfg["table"], mod.RadialSeriesTable)
        np.testing.assert_allclose(cfg["lob_centers"], dm.DEFAULT_LOB_CENTERS)

    def test_ell_max_3_is_accepted(self):
        cfg = mod.setup(make_options(dict(_base_options(), ell_max=3)))
        self.assertEqual(cfg["ell_max"], 3)

    def test_ell_max_outside_2_or_3_raises(self):
        for bad in (0, 1, 4):
            with self.assertRaises(ValueError):
                mod.setup(make_options(dict(_base_options(), ell_max=bad)))

    def test_explicit_table_path_is_loaded(self):
        default_path = pf.repo_root() / mod.DEFAULT_TABLE
        cfg = mod.setup(make_options(
            dict(_base_options(), table=str(default_path))))
        self.assertEqual(Path(cfg["table"].path), default_path)

    def test_missing_bin_index_or_r_perp_fails_loudly(self):
        for missing in ("bin_index", "r_perp"):
            entries = {k: v for k, v in _base_options().items()
                      if k != missing}
            with self.assertRaises(Exception, msg=f"missing {missing}"):
                mod.setup(make_options(entries))


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestExecuteAgainstExactMassIntegral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = datablock_from_dump(
            DUMP_DIR, dm,
            extra={("miscentering", "f_mis"): dm.F_MIS_DEFAULT,
                  ("miscentering", "tau_mis"): dm.TAU_MIS_DEFAULT})

    def test_execute_writes_the_documented_output_keys(self):
        cfg = mod.setup(make_options(_base_options()))
        rc = mod.execute(self.block, cfg)
        self.assertEqual(rc, 0)
        for key in ("vals", "norm", "y_eff", "mu2", "mu3"):
            self.assertTrue(self.block.has_value(mod.OUTPUT_SECTION, key),
                            f"missing output key {key}")
        vals = self.block[mod.OUTPUT_SECTION, "vals"]
        self.assertEqual(vals.size, BIN_INDEX.size * R_PERP.size)
        self.assertTrue(np.all(np.isfinite(vals)))
        self.assertTrue(np.all(vals > 0.0))

    def test_execute_matches_the_exact_fixed_gl_mass_integral(self):
        # Same reference validate_radial_series.py's own "check 3" uses:
        # the exact fixed-GL mass sum over the identical fixed-c profile
        # family, at its established ell<=2 tolerance -- isolates the
        # series-truncation error from the documented (and separately
        # tracked) fixed-c-vs-production shape difference.
        cfg = mod.setup(make_options(_base_options()))
        mod.execute(self.block, cfg)
        vals = np.asarray(self.block[mod.OUTPUT_SECTION, "vals"]
                          ).reshape(BIN_INDEX.size, R_PERP.size)

        source = dm.DataBlockSource(self.block)
        weights = dm.MassZWeights(
            source, n_lnm=cfg["n_lnm"], n_z=cfg["n_z"],
            zt_lo=cfg["zt_low"], zt_hi=cfg["zt_high"],
            lnm_lo=cfg["lnm_low"], lnm_hi=cfg["lnm_high"], include_sci=True)
        rho_ref = source.scalar("halomodel", "rho_m_ref")
        lob = cfg["lob_centers"]
        table = cfg["table"]
        src_tab = pf.MisTable()

        worst = 0.0
        for b in range(BIN_INDEX.size):
            r_mis = dm.TAU_MIS_DEFAULT * float(dm.R_lambda(lob[b % lob.size]))
            exact = exact_mass_integral(weights, table, src_tab, r_mis, b,
                                        dm.F_MIS_DEFAULT, rho_ref,
                                        use_source_table=False)
            dev = np.max(np.abs(vals[b] / exact - 1.0))
            worst = max(worst, dev)
        self.assertLess(worst, TRUNC_TOL_ELL2,
                        f"ell<=2 truncation {worst:.2e} exceeds "
                        f"{TRUNC_TOL_ELL2:.2e}")

    def test_physical_density_flag_is_implemented_and_changes_the_answer(self):
        # CLAUDE.md: shear1h_radial_series DOES implement
        # one_halo_physical_density (the (1+z)^2 identity), unlike the
        # explicit/max Python mirrors.
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
        self.assertFalse(np.allclose(on_vals, off_vals))

    def test_missing_miscentering_section_fails_loudly(self):
        bare = datablock_from_dump(DUMP_DIR, dm)
        cfg = mod.setup(make_options(_base_options()))
        with self.assertRaises(Exception):
            mod.execute(bare, cfg)

    def test_cleanup_is_a_no_op(self):
        self.assertEqual(mod.cleanup({}), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
