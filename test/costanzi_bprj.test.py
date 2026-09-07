#!/usr/bin/env python3
"""Costanzi-2026 B_prj(R) model (src/pipelines/systematics/costanzi_bprj/python).

Pins the closed form at R = R0, the R -> 0 / R >> R0 power-law limits, golden
values shared with costanzi_bprj.test.cc (transcribed independently from the
App. C formula of arXiv:2604.05833), the values-file DataBlock loader, the
gamma > 0 guard, the wall evaluation bprj_wall(block, R) (order and values)
and the CosmoSIS module that publishes lob_centers / zob_centers.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))

from systematics.costanzi_bprj.python import costanzi_bprj as cb  # noqa: E402
from systematics.costanzi_bprj.python.costanzi_bprj import (  # noqa: E402
    PARAM_NAMES,
    CostanziBprj,
    bprj_wall,
)

LOB, Z = 40.0, 0.3
# (R, lob = 40, z = 0.3) -> B, paper formula transcribed independently.
GOLDEN = {
    "sigma": {0.5: 1.091982576562725, 1.0: 1.091255050071865,
              3.0: 1.0581193649579876},
    "dsigma": {0.5: 1.0031265151008693, 1.0: 1.022544206891198,
               3.0: 1.105348390209914},
}


def _block_with(model, section="costanzi_bprj", **arrays):
    from cosmosis.datablock import DataBlock

    block = DataBlock()
    for name in PARAM_NAMES:
        block[section, name] = getattr(model, name)
    for key, val in arrays.items():
        block[section, key] = np.asarray(val, dtype=float)
    return block


class TestCostanziBprj(unittest.TestCase):
    def test_r0_is_r_lambda_times_1pz(self):
        self.assertAlmostEqual(CostanziBprj.r0(LOB, Z), 1.082319169622435, places=14)
        self.assertEqual(CostanziBprj.r0(100.0, 0.0), 1.0)

    def test_golden_values(self):
        for name, pins in GOLDEN.items():
            model = getattr(CostanziBprj, name)()
            for R, want in pins.items():
                np.testing.assert_allclose(model(R, LOB, Z), want, rtol=1e-12)
        # vectorized in R
        R = np.array([0.5, 1.0, 3.0])
        out = CostanziBprj.sigma()(R, LOB, Z)
        self.assertEqual(out.shape, (3,))
        np.testing.assert_allclose(out, [GOLDEN["sigma"][r] for r in R], rtol=1e-12)

    def test_closed_form_at_r0(self):
        for model in (CostanziBprj.sigma(), CostanziBprj.dsigma()):
            r0 = model.r0(LOB, Z)
            want = model.A * 2.0 ** ((model.beta - model.alpha) / model.gamma) + 1.0
            np.testing.assert_allclose(model(r0, LOB, Z), want, rtol=1e-14)

    def test_power_law_limits(self):
        model = CostanziBprj.sigma()
        r0 = model.r0(LOB, Z)
        self.assertEqual(float(model(0.0, LOB, Z)), 1.0)  # alpha > 0: no correction at R = 0
        xo, xi = 1e6, 1e-3
        np.testing.assert_allclose(model(xo * r0, LOB, Z) - 1.0,
                                   model.A * xo**model.beta, rtol=1e-9)
        np.testing.assert_allclose(model(xi * r0, LOB, Z) - 1.0,
                                   model.A * xi**model.alpha, rtol=1e-9)

    def test_from_datablock(self):
        block = _block_with(CostanziBprj.sigma())
        for name in PARAM_NAMES:
            block["bprj_dsigma", name] = getattr(CostanziBprj.dsigma(), name)
        self.assertEqual(CostanziBprj.from_datablock(block), CostanziBprj.sigma())
        self.assertEqual(CostanziBprj.from_datablock(block, "bprj_dsigma"),
                         CostanziBprj.dsigma())

    def test_gamma_guard(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            CostanziBprj(A=0.1, alpha=0.1, beta=-0.5, gamma=0.0)

    def test_bprj_wall_order_and_values(self):
        model = CostanziBprj.dsigma()
        lob = np.array([25.0, 37.5, 52.5, 130.0])
        zob = np.array([0.275, 0.425, 0.575])
        R = np.array([0.5, 1.0, 3.0])
        block = _block_with(model, lob_centers=lob, zob_centers=zob)
        B = bprj_wall(block, R)
        # z-major, richness fastest, radius fastest within a bin
        want = np.array([model(r, l, z) for z in zob for l in lob for r in R])
        self.assertEqual(B.shape, (zob.size * lob.size * R.size,))
        np.testing.assert_allclose(B, want, rtol=1e-14)
        self.assertGreater(B.max(), 1.0)
        with self.assertRaises(Exception):  # wall grid missing from the section
            bprj_wall(_block_with(model), R)

    def test_module_publishes_wall_grid(self):
        from cosmosis.datablock import DataBlock, option_section

        # defaults: the DES Y3 wall
        block = DataBlock()
        cb.execute(block, cb.setup(DataBlock()))
        np.testing.assert_allclose(block["costanzi_bprj", "lob_centers"],
                                   cb.DEFAULT_LOB_CENTERS)
        np.testing.assert_allclose(block["costanzi_bprj", "zob_centers"],
                                   cb.DEFAULT_ZOB_CENTERS)
        # explicit ini vectors
        options = DataBlock()
        options[option_section, "lob_centers"] = np.array([30.0, 60.0])
        options[option_section, "zob_centers"] = np.array([0.3])
        block = DataBlock()
        cb.execute(block, cb.setup(options))
        np.testing.assert_allclose(block["costanzi_bprj", "lob_centers"], [30.0, 60.0])
        np.testing.assert_allclose(block["costanzi_bprj", "zob_centers"], [0.3])


if __name__ == "__main__":
    unittest.main()
