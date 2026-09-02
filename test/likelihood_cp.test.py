#!/usr/bin/env python3
"""y3_buzzard/likelihood_cp.py: max-model mode (shear_max_section) and the
Costanzi-2026 B_prj(R) option (is_b_proj_costanzi26).

Synthetic DataBlock + DV file, dump-free. Pins: the max model closes at
data (logL = 0), the B_prj-multiplied theory reproduces an independent
CostanziBprj evaluation on the wall the costanzi_bprj section defines,
[costanzi_bprj] is read per sample (A = 0 -> B = 1 -> closure), the flag
without a max section or without the wall grid is an error, and the
default 1h + prj path still closes.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from cosmosis.datablock import DataBlock, option_section

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))
from systematics.costanzi_bprj.python.costanzi_bprj import (  # noqa: E402
    PARAM_NAMES,
    CostanziBprj,
)

N_BINS, N_R = 12, 10
R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
LOB_CENTERS = np.array([25.0, 37.5, 52.5, 130.0])
ZOB_CENTERS = np.array([0.275, 0.425, 0.575])


def _load_likelihood_cp():
    spec = importlib.util.spec_from_file_location(
        "likelihood_cp", REPO / "y3_buzzard" / "likelihood_cp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestLikelihoodCpMaxModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lk = _load_likelihood_cp()
        rng = np.random.default_rng(1)
        cls.NC = rng.uniform(50.0, 2000.0, N_BINS)
        cls.shear = rng.uniform(1e-3, 1e-1, N_BINS * N_R)  # per-cluster gamma_t
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dv = Path(cls.tmp.name) / "dv.npz"
        np.savez(cls.dv, data_NC=cls.NC, invcov_NC=np.ones(N_BINS),
                 data_Shear=cls.shear,
                 invcov_Shear=np.full(N_BINS * N_R, 1.0e4))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _options(self, **kw):
        o = DataBlock()
        o[option_section, "filename"] = str(self.dv)
        o[option_section, "num_counts_section"] = "numcounts"
        for k, v in kw.items():
            o[option_section, k] = v
        return o

    def _block(self, bprj=None, wall=True):
        b = DataBlock()
        b["numcounts", "vals"] = self.NC
        # the module publishes the N_i-weighted integral
        b["shear1h2h_max", "vals"] = self.shear * np.repeat(self.NC, N_R)
        if bprj is not None:
            for k in PARAM_NAMES:
                b["costanzi_bprj", k] = getattr(bprj, k)
            if wall:  # what the costanzi_bprj module stage publishes
                b["costanzi_bprj", "lob_centers"] = LOB_CENTERS
                b["costanzi_bprj", "zob_centers"] = ZOB_CENTERS
        return b

    def test_max_model_closes_at_data(self):
        cfg = self.lk.setup(self._options(shear_max_section="shear1h2h_max"))
        self.assertFalse(cfg["is_b_proj_costanzi26"])
        b = self._block()
        self.lk.execute(b, cfg)
        self.assertAlmostEqual(b["likelihoods", "likelihoods_like"], 0.0, places=8)

    def test_b_proj_multiplies_max_model(self):
        bprj = CostanziBprj.dsigma()
        cfg = self.lk.setup(self._options(shear_max_section="shear1h2h_max",
                                          is_b_proj_costanzi26=True))
        b = self._block(bprj)
        self.lk.execute(b, cfg)
        # independent evaluation: bins z-major (richness fast), radius fastest
        B = np.array([bprj(r, lob, z) for z in ZOB_CENTERS
                      for lob in LOB_CENTERS for r in R_PERP])
        theory = self.shear * B
        want = -0.5 * np.sum((self.shear - theory) ** 2 * 1.0e4)
        self.assertLess(want, -1.0)  # B != 1: the correction actually bites
        np.testing.assert_allclose(b["likelihoods", "likelihoods_like"], want,
                                   rtol=1e-10)

    def test_b_proj_params_read_per_sample(self):
        cfg = self.lk.setup(self._options(shear_max_section="shear1h2h_max",
                                          is_b_proj_costanzi26=True))
        b = self._block(CostanziBprj(A=0.0, alpha=4.11, beta=0.18, gamma=1.82))
        self.lk.execute(b, cfg)  # A = 0 -> B = 1 -> closure
        self.assertAlmostEqual(b["likelihoods", "likelihoods_like"], 0.0, places=8)

    def test_b_proj_requires_wall_grid_in_block(self):
        cfg = self.lk.setup(self._options(shear_max_section="shear1h2h_max",
                                          is_b_proj_costanzi26=True))
        with self.assertRaises(Exception):  # costanzi_bprj module not run
            self.lk.execute(self._block(CostanziBprj.dsigma(), wall=False), cfg)

    def test_b_proj_requires_max_model(self):
        with self.assertRaises(ValueError):
            self.lk.setup(self._options(is_b_proj_costanzi26=True))

    def test_default_1h_prj_path_unchanged(self):
        cfg = self.lk.setup(self._options(shear_1h_section="s1h",
                                          shear_prj_section="sprj"))
        b = DataBlock()
        b["numcounts", "vals"] = self.NC
        half = 0.5 * self.shear
        b["s1h", "vals"] = half * np.repeat(self.NC, N_R)
        b["sprj", "vals"] = half
        self.lk.execute(b, cfg)
        self.assertAlmostEqual(b["likelihoods", "likelihoods_like"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
