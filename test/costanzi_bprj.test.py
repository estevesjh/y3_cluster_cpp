#!/usr/bin/env python3
"""Costanzi-2026 B_prj(R) model (src/pipelines/systematics/costanzi_bprj/python).

Pins the closed form at R = R0, the R -> 0 / R >> R0 power-law limits, golden
values shared with costanzi_bprj.test.cc (transcribed independently from the
App. C formula of arXiv:2604.05833), the values-file DataBlock loader and the
gamma > 0 guard.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))

from systematics.costanzi_bprj.python.costanzi_bprj import (  # noqa: E402
    PARAM_NAMES,
    CostanziBprj,
)

LOB, Z = 40.0, 0.3
# (R, lob = 40, z = 0.3) -> B, paper formula transcribed independently.
GOLDEN = {
    "sigma": {0.5: 1.091982576562725, 1.0: 1.091255050071865,
              3.0: 1.0581193649579876},
    "dsigma": {0.5: 1.0031265151008693, 1.0: 1.022544206891198,
               3.0: 1.105348390209914},
}


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
        from cosmosis.datablock import DataBlock

        block = DataBlock()
        for section, preset in (("costanzi_bprj", CostanziBprj.sigma()),
                                ("bprj_dsigma", CostanziBprj.dsigma())):
            for name in PARAM_NAMES:
                block[section, name] = getattr(preset, name)
        self.assertEqual(CostanziBprj.from_datablock(block), CostanziBprj.sigma())
        self.assertEqual(CostanziBprj.from_datablock(block, "bprj_dsigma"),
                         CostanziBprj.dsigma())

    def test_gamma_guard(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            CostanziBprj(A=0.1, alpha=0.1, beta=-0.5, gamma=0.0)


if __name__ == "__main__":
    unittest.main()
