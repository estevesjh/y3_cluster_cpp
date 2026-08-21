"""The compiled photo-z width table sigma_z(z) — z_kernel_data.hh reader.

The C++ P operators and the projection evaluators consume the 120-node
sigma_z(z) table compiled into src/models/z_kernel_data.hh (the single
source of truth since the data/z_kernel text files were retired).
Python implementations parse the same header at import so the two
languages cannot drift apart.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "src" / "models" / "z_kernel_data.hh").is_file():
            return p
    raise FileNotFoundError("z_kernel: cannot locate z_kernel_data.hh")


@lru_cache(maxsize=1)
def z_kernel_table():
    """Return (z, sigma) arrays parsed from z_kernel_data.hh."""
    text = (_repo_root() / "src" / "models" / "z_kernel_data.hh").read_text()
    out = []
    for name in ("Z_KERNEL_Z", "Z_KERNEL_SIGMA"):
        m = re.search(name + r"[^{]*\{([^}]*)\}", text, re.S)
        if not m:
            raise ValueError(f"z_kernel: array {name} not found")
        out.append(np.array([float(t) for t in
                             re.findall(r"[-0-9.eE+]+", m.group(1))]))
    z, sigma = out
    if z.size != sigma.size or z.size == 0:
        raise ValueError("z_kernel: parsed arrays inconsistent")
    return z, sigma


def sigma_z(z):
    """Clamped linear interpolation of sigma_z(z) (Interp1D::clamp)."""
    zt, st = z_kernel_table()
    return np.interp(np.clip(np.asarray(z, dtype=float), zt[0], zt[-1]),
                     zt, st)
