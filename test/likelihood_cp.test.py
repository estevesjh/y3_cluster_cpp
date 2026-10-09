#!/usr/bin/env python3
"""Exercise the max-model and default paths in likelihood_cp.py.

Synthetic DataBlock + DV file, dump-free. The max model and the default
1h + projection path both close on matching synthetic data.
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from cosmosis.datablock import DataBlock, option_section

REPO = Path(__file__).resolve().parents[1]

N_BINS, N_R = 12, 10
def _load_likelihood_cp():
    spec = importlib.util.spec_from_file_location(
        "likelihood_cp", REPO / "src" / "pipelines" / "buzzard" / "likelihoods" / "likelihood_cp.py")
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

    def _block(self):
        b = DataBlock()
        b["numcounts", "vals"] = self.NC
        # the module publishes the N_i-weighted integral
        b["shear1h2h_max", "vals"] = self.shear * np.repeat(self.NC, N_R)
        return b

    def test_max_model_closes_at_data(self):
        cfg = self.lk.setup(self._options(shear_max_section="shear1h2h_max"))
        b = self._block()
        self.lk.execute(b, cfg)
        self.assertAlmostEqual(b["likelihoods", "likelihoods_like"], 0.0, places=8)

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
