#!/usr/bin/env python3
"""Shared test-only helper: replay a cosmosis test-sampler dump into a real
``cosmosis.datablock.DataBlock``.

The ``des_y3`` python 0d backends' ``execute(block, cfg)`` entry points
accept a live DataBlock (they wrap it themselves via
``shared.datablock_models.DataBlockSource``/``sel_kernels._read_mor`` etc,
not the offline ``DumpSource``). To exercise the REAL ``setup()``/
``execute()`` CosmoSIS contract -- not just the pure ``compute_*`` helpers
``test/des_y3_pipeline.test.py`` already covers -- this module re-publishes
every ``values.txt`` scalar and ``<key>.txt`` array of a
``cosmosis-models/*_output`` dump directory into a fresh, real DataBlock,
the same way CosmoSIS itself would have populated it in the original run.

Not a ``*.test.py`` file itself (no ``Test*`` classes), so it is not
collected by ``pytest test/*.test.py`` / ctest and is safe to import from
multiple test modules.
"""
from __future__ import annotations

import os
import sys

import numpy as np


def _real_cosmosis_datablock():
    """Return the REAL ``cosmosis.datablock`` module, healing around any
    incomplete stub another test file installed into ``sys.modules``.

    ``test/explicit_grid_core.test.py`` (and others) install a minimal
    ``cosmosis``/``cosmosis.datablock`` stub -- exposing only
    ``option_section``/``names`` -- via
    ``sys.modules.setdefault(...)`` at module import time, with no
    teardown, so that it can import ``sel_function.py`` without a live
    CosmoSIS environment. When the full suite runs in one pytest process
    and that file is collected before this one, the stub wins the
    ``setdefault`` race and every later ``from cosmosis.datablock import
    DataBlock`` in THIS file raises ``ImportError`` even though real
    cosmosis is installed and otherwise importable. Since this repo's
    ``sel_function.py``/``prj_params.py`` only ever read
    ``option_section``/``names`` from the module (identical values on
    both the stub and the real module), swapping the real module back in
    here is safe for every other consumer already holding a reference.
    """
    mod = sys.modules.get("cosmosis.datablock")
    if mod is not None and hasattr(mod, "DataBlock"):
        return mod
    sys.modules.pop("cosmosis.datablock", None)
    sys.modules.pop("cosmosis", None)
    import cosmosis.datablock as real_mod
    return real_mod


def datablock_from_dump(dump_dir, dm, extra=None):
    """Return a real ``DataBlock`` populated from a dump directory.

    ``dm`` is the caller's already-imported ``shared.datablock_models``
    module (passed in rather than imported here so every test module keeps
    its own single import of the shared layer). ``extra`` is an optional
    ``{(section, key): value}`` mapping applied AFTER the dump replay, for
    sections the dump legitimately does not carry (e.g. ``miscentering``,
    absent from ``real_pipeline_extract_output`` -- see
    ``des_y3_pipeline.test.py``'s ``TestShear1hGl``, which works around the
    same gap by constructing ``MisMixtureProfile`` with the in-code
    ``F_MIS_DEFAULT``/``TAU_MIS_DEFAULT`` constants directly).
    """
    DataBlock = _real_cosmosis_datablock().DataBlock

    src = dm.DumpSource(str(dump_dir))
    block = DataBlock()
    for section_dir in sorted(os.listdir(dump_dir)):
        full = os.path.join(dump_dir, section_dir)
        if not os.path.isdir(full):
            continue
        values_path = os.path.join(full, "values.txt")
        if os.path.exists(values_path):
            with open(values_path) as fh:
                for line in fh:
                    if "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    try:
                        block[section_dir, key] = float(val)
                    except ValueError:
                        continue
        for fname in sorted(os.listdir(full)):
            if not fname.endswith(".txt") or fname == "values.txt":
                continue
            key = fname[:-len(".txt")]
            arr = src.array(section_dir, key)
            block[section_dir, key] = np.asarray(arr, dtype=float)

    if extra:
        for (section, key), value in extra.items():
            block[section, key] = value
    return block


def make_options(entries):
    """Build a real cosmosis ``options`` DataBlock under ``option_section``.

    ``entries`` maps option name -> value. Arrays of integer dtype publish
    through ``put_int_array_1d`` (so ``sel_function._read_array``'s
    double-then-int fallback is genuinely exercised for e.g.
    ``bin_index``); float arrays publish as double arrays. Scalars publish
    as bool/string/int/double by Python type.
    """
    real = _real_cosmosis_datablock()
    DataBlock, option_section = real.DataBlock, real.option_section

    opts = DataBlock()
    for name, value in entries.items():
        if isinstance(value, bool):
            opts.put_bool(option_section, name, value)
        elif isinstance(value, str):
            opts.put_string(option_section, name, value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if np.issubdtype(arr.dtype, np.integer):
                opts.put_int_array_1d(option_section, name,
                                      arr.astype(np.int32))
            else:
                opts.put_double_array_1d(option_section, name,
                                         arr.astype(float))
        elif isinstance(value, int):
            opts.put_int(option_section, name, value)
        else:
            opts.put_double(option_section, name, float(value))
    return opts
