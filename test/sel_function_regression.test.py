#!/usr/bin/env python3
"""Compare the refactored selection pipeline with its pre-refactor version.

The test executes both implementations on the same in-memory CosmoSIS-like
datablock.  The previous implementation is loaded directly from the Git
baseline so that this test does not copy, and therefore cannot silently drift
from, the old numerical pipeline.

Two properties are checked:

* ``sel_function/S_stack`` agrees to a tight floating-point tolerance;
* after one warm-up execution has paid the Numba compilation cost, the
  refactored implementation is no more than 50% slower by median wall time.

This is intentionally a small production-shaped test rather than a unit test
of private formulas.  It covers setup, datablock reads, the fast HOD kernel,
the analytical PLOB CDF edge evaluation, the PHOD x PLOB_LTR contraction, and
the output datavector consumed by the next pipeline stages.
"""
from __future__ import annotations

import contextlib
import io
import statistics
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PIPELINES = REPO / "src" / "pipelines"
BASELINE_COMMIT = "80b2fcf"
BASELINE_PATH = "src/modules/sel_function/sel_function.py"
TIMING_SLOWDOWN_LIMIT = 1.50

if str(PIPELINES) not in sys.path:
    sys.path.insert(0, str(PIPELINES))

from cosmology.prj_params import PrjParams  # noqa: E402
from shared import sel_kernels  # noqa: E402

current_sel_function = sel_kernels.load()


class MemoryBlock:
    """Minimal live-DataBlock surface used by both pipeline versions."""

    def __init__(self):
        self.values = {}

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def has_value(self, section, name):
        return (section, name) in self.values


class Options:
    """CosmoSIS option reader for the small deterministic regression grid."""

    def __init__(self, values):
        self.values = values

    def _value(self, name):
        return self.values[name]

    def get_double_array_1d(self, _section, name):
        value = self._value(name)
        if np.asarray(value).ndim == 0:
            raise TypeError(f"{name} is a scalar option")
        return np.asarray(value, dtype=float)

    def get_int_array_1d(self, _section, name):
        value = self._value(name)
        if np.asarray(value).ndim == 0:
            raise TypeError(f"{name} is a scalar option")
        return np.asarray(value, dtype=int)

    def get_double(self, _section, name):
        return float(self._value(name))

    def get_int(self, _section, name):
        return int(self._value(name))


def make_options():
    """Return one small but production-shaped 12-bin selection wall."""
    richness_min = np.tile([20.0, 30.0, 45.0, 60.0], 3)
    richness_max = np.tile([30.0, 45.0, 60.0, 200.0], 3)
    redshift_min = np.repeat([0.20, 0.35, 0.50], 4)
    redshift_max = np.repeat([0.35, 0.50, 0.65], 4)
    return Options({
        "lam_min": richness_min,
        "lam_max": richness_max,
        "zob_min": redshift_min,
        "zob_max": redshift_max,
        "sigma_z": np.full(12, 0.03),
        "zt_low": 0.05,
        "zt_high": 0.80,
        "lnm_low": np.log(1.0e13),
        "lnm_high": np.log(9.0e15),
        # These are deliberately smaller than the production grid so the
        # regression test stays quick while still exercising every tensor.
        "n_lnm": 48,
        "n_z": 16,
        "n_z_shared": 24,
        "L_z": 6.0,
        "L_lam": 6.0,
        "N_q": 20,
    })


def make_block():
    """Create the same sampled inputs for both implementations."""
    block = MemoryBlock()
    block["cluster_mor", "log10_Mmin"] = 13.8
    block["cluster_mor", "log10_M1"] = 14.5
    block["cluster_mor", "alpha"] = 1.1
    block["cluster_mor", "epsilon"] = -0.2
    block["cluster_mor", "sigma_lambda"] = 0.35
    block["cluster_mor", "z_pivot"] = 0.45

    # Publish the frozen projection table exactly as the prj_params module
    # does in a real sample.  This ensures both paths read the same inputs,
    # rather than comparing two different fallback mechanisms.
    for name, values in PrjParams.default().as_dict().items():
        block["plob_ltr_params", name] = np.asarray(values, dtype=float)
    return block


def load_previous_pipeline():
    """Load the exact pre-refactor module from the repository baseline.

    Numba's on-disk cache cannot use an in-memory module, so only the cache
    flag is disabled in this test copy.  The numerical source is otherwise
    unchanged.  The old package name is provided as a compatibility alias
    because the repository moved that module during the refactor.
    """
    try:
        source = subprocess.check_output(
            ["git", "show", f"{BASELINE_COMMIT}:{BASELINE_PATH}"],
            cwd=REPO,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise unittest.SkipTest(
            "selection regression requires Git baseline "
            f"{BASELINE_COMMIT}: {error}"
        ) from error

    # The old source imports this name; the implementation it points to is
    # the same frozen PrjParams table used by the current source.
    import cosmology.prj_params as prj_params

    legacy_package = types.ModuleType("y3_buzzard")
    legacy_package.prj_params = prj_params
    sys.modules.setdefault("y3_buzzard", legacy_package)
    sys.modules.setdefault("y3_buzzard.prj_params", prj_params)

    module_name = "_previous_sel_function_80b2fcf"
    module = types.ModuleType(module_name)
    module.__file__ = str(REPO / BASELINE_PATH)
    module.__package__ = ""
    sys.modules[module_name] = module

    # The baseline used cache=True.  The test loads it from a string, so its
    # source location is not a normal importable file for Numba's cache.
    source = source.replace("cache=True", "cache=False")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def execute_once(module, config):
    """Run one sample and return elapsed seconds plus the output block."""
    block = make_block()
    output = io.StringIO()
    start = time.perf_counter()
    # Zero-width HOD cells use zero true-richness nodes.  Both the baseline
    # and current projection helper consequently evaluate log(0) in a value
    # that is never used; suppress that expected NumPy warning in the report.
    with np.errstate(divide="ignore", invalid="ignore"), \
            contextlib.redirect_stdout(output):
        status = module.execute(block, config)
    elapsed = time.perf_counter() - start
    if status != 0:
        raise AssertionError(f"sel_function returned status {status}")
    return elapsed, block


class TestSelectionPipelineRegression(unittest.TestCase):
    """Numerical and performance guard for the selection-function refactor."""

    @classmethod
    def setUpClass(cls):
        cls.previous_sel_function = load_previous_pipeline()
        options = make_options()
        cls.current_config = current_sel_function.setup(options)
        cls.previous_config = cls.previous_sel_function.setup(options)

    def test_output_matches_previous_pipeline(self):
        # The first calls also compile the two independent Numba modules.
        _, previous_block = execute_once(
            self.previous_sel_function, self.previous_config)
        _, current_block = execute_once(
            current_sel_function, self.current_config)

        previous_grid = previous_block["sel_function", "S_stack"]
        current_grid = current_block["sel_function", "S_stack"]
        np.testing.assert_allclose(
            current_block["sel_function", "lnM"],
            previous_block["sel_function", "lnM"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            current_block["sel_function", "z"],
            previous_block["sel_function", "z"],
            rtol=0.0,
            atol=0.0,
        )
        # These are the new explicit wall datavectors consumed by bsel and
        # the shear consumers.  They are derived from the same configured
        # bin edges, so check them independently of the legacy S_stack.
        np.testing.assert_array_equal(
            current_block["sel_function", "lambda_edges"],
            np.array([20.0, 30.0, 45.0, 60.0, 200.0]),
        )
        np.testing.assert_allclose(
            current_block["sel_function", "lambda_centres"],
            np.array([25.0, 37.5, 52.5, 130.0]),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            current_grid,
            previous_grid,
            rtol=2.0e-6,
            atol=1.0e-12,
        )

        difference = np.abs(current_grid - previous_grid)
        # Relative error on values below 1e-6 of the peak is dominated by
        # the absolute floor and is not useful as a precision diagnostic.
        relevant_floor = 1.0e-6 * np.max(np.abs(previous_grid))
        scale = np.maximum(np.abs(previous_grid), relevant_floor)
        print(
            "[sel_function regression] max absolute difference = "
            f"{difference.max():.3e}, max scaled difference = "
            f"{np.max(difference / scale):.3e}"
        )

    def test_runtime_is_similar_to_previous_pipeline(self):
        # Warm-up keeps one-time Numba compilation out of the benchmark.  The
        # same setup/configuration is then reused, matching a CosmoSIS module
        # instance processing successive MCMC samples.
        execute_once(self.previous_sel_function, self.previous_config)
        execute_once(current_sel_function, self.current_config)

        previous_times = [
            execute_once(self.previous_sel_function, self.previous_config)[0]
            for _ in range(3)
        ]
        current_times = [
            execute_once(current_sel_function, self.current_config)[0]
            for _ in range(3)
        ]
        previous_median = statistics.median(previous_times)
        current_median = statistics.median(current_times)
        ratio = current_median / max(previous_median, np.finfo(float).tiny)
        print(
            "[sel_function timing] previous median = "
            f"{previous_median * 1.0e3:.2f} ms, current median = "
            f"{current_median * 1.0e3:.2f} ms, ratio = {ratio:.3f}"
        )
        self.assertLessEqual(
            ratio,
            TIMING_SLOWDOWN_LIMIT,
            msg=(
                "refactored sel_function is more than "
                f"{TIMING_SLOWDOWN_LIMIT:.0%} slower than the baseline"
            ),
        )


if __name__ == "__main__":
    unittest.main()
