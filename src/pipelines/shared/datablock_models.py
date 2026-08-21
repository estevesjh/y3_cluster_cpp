"""Convention-exact numpy replicas of the shared C++ model layer.

Every class here mirrors one struct under ``src/models/`` (or the SelGLCore
weight builder in ``src/models/n_operator_sel_gl_t.hh``), including its
interpolation scheme, so that Python implementations under
``src/pipelines/des_y3`` compose *the same* numbers as the production
fixed-GL modules:

    HMF          <-> src/models/hmf_t.hh          (bilinear + clamp)
    DVDoDz       <-> src/models/dv_do_dz_t.hh     (linear d_a interp)
    omega_z_des  <-> src/models/omega_z_des.hh    (piecewise polynomial)
    SelStack     <-> src/models/sel_function_t.hh (bilinear, 0 outside)
    SigmaCritInv <-> lensing_weights.hh load_sigma_crit_inv (linear + clamp)
    MassZWeights <-> nosel_gl_detail::SelGLCore   (fixed-GL z contraction)

Two input adapters expose one read interface for both runtime and offline
work: ``DataBlockSource`` wraps a live ``cosmosis`` DataBlock;
``DumpSource`` wraps a ``test``-sampler ``save_dir`` dump so validation
scripts replay exactly what a real sample saw.

Convention notes replicated deliberately (do not "fix" these here):

- HMF_t's mass axis is ``ln(m_h * (omega_m - omega_nu))`` and production
  callers (SelGLCore, the P operators) query it with the *same* lnM
  coordinate used for the selection tensor. This module keeps that exact
  pairing; see the unit-conventions section of CLAUDE.md.
- The HMF nuisance factor is ``hmf_s * (log10(M) - 13.8124426028) + hmf_q``
  evaluated at the query lnM, exactly as hmf_t.hh.
- DV_DO_DZ is in (Mpc/h)^3 per steradian per unit z, with d_a in Mpc from
  the ``distances`` section and the 2997.92 Hubble-distance constant.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

GL_CACHE = {}


def gl_nodes(a, b, n):
    """Fixed Gauss-Legendre nodes/weights on [a, b] (p_op_detail::gl_nodes)."""
    key = int(n)
    if key not in GL_CACHE:
        GL_CACHE[key] = np.polynomial.legendre.leggauss(key)
    t, w = GL_CACHE[key]
    x = 0.5 * (b - a) * t + 0.5 * (b + a)
    return x, 0.5 * (b - a) * w


# ---------------------------------------------------------------------------
# Input adapters
# ---------------------------------------------------------------------------

class DataBlockSource:
    """Read adapter over a live cosmosis DataBlock."""

    def __init__(self, block):
        self.block = block

    def array(self, section, key):
        return np.asarray(self.block[section, key], dtype=float)

    def scalar(self, section, key):
        return float(self.block[section, key])

    def has(self, section, key):
        return self.block.has_value(section, key)


class DumpSource:
    """Read adapter over a cosmosis test-sampler save_dir directory.

    Scalars live in each section's ``values.txt`` (``name = value`` rows);
    arrays are one ``<key>.txt`` per key. 3-D arrays (``s_stack``) carry a
    ``# shape = (...)`` header line that we use to reshape.
    """

    def __init__(self, dirpath):
        self.dir = dirpath
        self._values = {}

    def _section_values(self, section):
        if section not in self._values:
            vals = {}
            path = os.path.join(self.dir, section, "values.txt")
            if os.path.exists(path):
                for line in open(path):
                    if "=" in line:
                        k, v = line.split("=", 1)
                        try:
                            vals[k.strip().lower()] = float(v)
                        except ValueError:
                            pass
            self._values[section] = vals
        return self._values[section]

    def array(self, section, key):
        path = os.path.join(self.dir, section, key.lower() + ".txt")
        shape = None
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                if "shape" in line:
                    shape = tuple(
                        int(t) for t in
                        line.split("=")[1].strip(" ()\n").split(",") if t.strip())
        out = np.loadtxt(path)
        return out.reshape(shape) if shape else out

    def scalar(self, section, key):
        return self._section_values(section)[key.lower()]

    def has(self, section, key):
        if key.lower() in self._section_values(section):
            return True
        return os.path.exists(
            os.path.join(self.dir, section, key.lower() + ".txt"))


# ---------------------------------------------------------------------------
# Cosmology / geometry terms
# ---------------------------------------------------------------------------

class EZ:
    """E(z) = sqrt(omega_m (1+z)^3 + omega_k (1+z)^2 + omega_lambda)."""

    def __init__(self, source):
        self.omega_m = source.scalar("cosmological_parameters", "omega_m")
        self.omega_k = source.scalar("cosmological_parameters", "omega_k")
        self.omega_lambda = source.scalar("cosmological_parameters",
                                          "omega_lambda")

    def __call__(self, z):
        zp1 = 1.0 + np.asarray(z, dtype=float)
        return np.sqrt(self.omega_m * zp1**3 + self.omega_k * zp1**2 +
                       self.omega_lambda)


class DVDoDz:
    """dV/dOmega/dz in (Mpc/h)^3 — replica of DV_DO_DZ_t."""

    def __init__(self, source):
        self._z = source.array("distances", "z")
        self._d_a = source.array("distances", "d_a")
        self._h = source.scalar("cosmological_parameters", "h0")
        self._ez = EZ(source)

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        da_h = np.interp(z, self._z, self._d_a) * self._h
        return 2997.92 * (1.0 + z)**2 * da_h * da_h / self._ez(z)


def _horner(coeffs, x):
    out = np.zeros_like(x)
    for c in coeffs:
        out = c + x * out
    return out


def omega_z_des(z):
    """Effective survey area Omega(z) in rad^2 — replica of OMEGA_Z_DES."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    c1 = [0.0, 0.0, 0.0, -0.00262353, 0.01940118, 0.45133063]
    c2 = [1.33647377e4, 1.35291046e3, -1.26204891e2,
          -2.83454918e1, -2.26465905, 3.84958753e-1]
    c3 = [0.0, 0.0, -1.88101967, 4.8071839, -4.11424324, 1.18196785]
    m1 = z < 0.504
    m2 = (~m1) & (z < 0.7)
    m3 = ~(m1 | m2)
    out[m1] = _horner(c1, z[m1])
    out[m2] = _horner(c2, z[m2] - 0.6)
    out[m3] = _horner(c3, z[m3])
    return out


# ---------------------------------------------------------------------------
# Halo mass function
# ---------------------------------------------------------------------------

class HMF:
    """dn/dlnM with the HMF_t axis and nuisance conventions.

    The x-axis is ``ln(m_h * (omega_m - omega_nu))`` and the query
    coordinate is passed through unmodified — the exact HMF_t pairing used
    by every production caller in the fixed-GL path.
    """

    def __init__(self, source):
        m_h = source.array("mass_function", "m_h")
        omega_m = source.scalar("cosmological_parameters", "omega_m")
        omega_nu = source.scalar("cosmological_parameters", "omega_nu")
        self._lnm = np.log(m_h * (omega_m - omega_nu))
        self._z = source.array("mass_function", "z")
        dndlnmh = np.asarray(source.array("mass_function", "dndlnmh"),
                             dtype=float)
        if dndlnmh.shape != (self._z.size, self._lnm.size):
            dndlnmh = dndlnmh.reshape(self._z.size, self._lnm.size)
        self._interp = RegularGridInterpolator(
            (self._z, self._lnm), dndlnmh, method="linear",
            bounds_error=False, fill_value=None)
        self._s = source.scalar("cluster_abundance", "hmf_s")
        self._q = source.scalar("cluster_abundance", "hmf_q")

    def __call__(self, lnM, z):
        lnM = np.asarray(lnM, dtype=float)
        z = np.asarray(z, dtype=float)
        lnM_c = np.clip(lnM, self._lnm[0], self._lnm[-1])
        z_c = np.clip(z, self._z[0], self._z[-1])
        lnM_b, z_b = np.broadcast_arrays(lnM_c, z_c)
        vals = self._interp(np.stack([z_b, lnM_b], axis=-1))
        # 0.4342944819 = log10(e); nuisance evaluated at the query lnM.
        nuis = self._s * (lnM * 0.4342944819 - 13.8124426028) + self._q
        return vals * nuis


# ---------------------------------------------------------------------------
# Selection tensor and Sigma_crit_inv
# ---------------------------------------------------------------------------

class SelStack:
    """Bilinear per-bin reader of sel_function/S_stack (SelFunction_t)."""

    def __init__(self, source):
        self.lnm = source.array("sel_function", "lnm")
        self.z = source.array("sel_function", "z")
        s_stack = np.asarray(source.array("sel_function", "s_stack"),
                             dtype=float)
        if s_stack.ndim != 3:
            s_stack = s_stack.reshape(-1, self.z.size, self.lnm.size)
        self.n_bins = s_stack.shape[0]
        self._interps = [
            RegularGridInterpolator((self.z, self.lnm), s_stack[b],
                                    method="linear", bounds_error=False,
                                    fill_value=0.0)
            for b in range(self.n_bins)
        ]

    def __call__(self, b, lnM, z):
        lnM_b, z_b = np.broadcast_arrays(np.asarray(lnM, dtype=float),
                                         np.asarray(z, dtype=float))
        return self._interps[b](np.stack([z_b, lnM_b], axis=-1))


class Bilinear2D:
    """Generic clamped-bilinear replica of make_Interp2D over a section.

    C++ convention: Interp2D(xs, ys, zs) with the value table stored
    rows = y, cols = x; queries are clamped to the domain.
    """

    def __init__(self, source, section, xkey, ykey, valkey, nan_fill=None):
        self._x = source.array(section, xkey)
        self._y = source.array(section, ykey)
        vals = np.asarray(source.array(section, valkey), dtype=float)
        if vals.shape != (self._y.size, self._x.size):
            vals = vals.reshape(self._y.size, self._x.size)
        if nan_fill is not None:
            vals = np.where(np.isfinite(vals), vals, float(nan_fill))
        self._interp = RegularGridInterpolator(
            (self._y, self._x), vals, method="linear",
            bounds_error=False, fill_value=None)

    def __call__(self, x, y):
        x = np.clip(np.asarray(x, dtype=float), self._x[0], self._x[-1])
        y = np.clip(np.asarray(y, dtype=float), self._y[0], self._y[-1])
        y_b, x_b = np.broadcast_arrays(y, x)
        return self._interp(np.stack([y_b, x_b], axis=-1))


class SigmaCritInv:
    """Source-averaged Sigma_crit^-1(z_lens), linear with edge clamping."""

    def __init__(self, source):
        self._z = source.array("average_sigma_crit_inv", "zlense")
        self._sci = source.array("average_sigma_crit_inv", "sci_average")

    def __call__(self, z):
        z = np.clip(np.asarray(z, dtype=float), self._z[0], self._z[-1])
        return np.interp(z, self._z, self._sci)


# ---------------------------------------------------------------------------
# Miscentering helpers (lensing_weights.hh mis_detail)
# ---------------------------------------------------------------------------

F_MIS_DEFAULT = 0.22
TAU_MIS_DEFAULT = 0.17
DEFAULT_LOB_CENTERS = (25.0, 37.5, 52.5, 130.0)


def R_lambda(lob):
    """R_lambda(lambda^ob) = (lambda^ob / 100)^0.2 [Mpc/h]."""
    return (np.asarray(lob, dtype=float) / 100.0) ** 0.2


def read_mis_param(source, key, default):
    try:
        return source.scalar("miscentering", key)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Fixed-GL z-marginalised mass weights (SelGLCore replica)
# ---------------------------------------------------------------------------

class MassZWeights:
    """Per-sample W_ij(lnM) on fixed GL nodes, plus arbitrary moments.

        W_ij(lnM) = int dz n(M,z) dV/dOmega/dz Omega(z) S_ij(lnM,z)
                    [ * Sigma_crit_inv(z), shear only ]

    Mirrors nosel_gl_detail::SelGLCore::build_weights: identical node
    placement (GL on [zt_lo, zt_hi] and [lnm_lo, lnm_hi]) and identical
    term composition, vectorised over (bin, lnM node, z node).
    """

    def __init__(self, source, *, n_lnm=96, n_z=64,
                 zt_lo, zt_hi, lnm_lo, lnm_hi, include_sci=False):
        self.lnm_x, self.lnm_w = gl_nodes(lnm_lo, lnm_hi, n_lnm)
        self.z_x, self.z_w = gl_nodes(zt_lo, zt_hi, n_z)

        hmf = HMF(source)
        dv = DVDoDz(source)
        sel = SelStack(source)
        self.n_bins = sel.n_bins

        zfac = self.z_w * dv(self.z_x) * omega_z_des(self.z_x)
        if include_sci:
            zfac = zfac * SigmaCritInv(source)(self.z_x)

        lnm_grid = self.lnm_x[:, None]
        z_grid = self.z_x[None, :]
        hmf_kq = hmf(lnm_grid, z_grid)                      # (n_lnm, n_z)
        self.W = np.empty((self.n_bins, self.lnm_x.size))
        for b in range(self.n_bins):
            s_kq = sel(b, lnm_grid, z_grid)                 # (n_lnm, n_z)
            self.W[b] = (hmf_kq * s_kq) @ zfac

    def norm(self):
        """int dlnM W_ij for every bin — the NumCountsSel observable."""
        return self.W @ self.lnm_w

    def moments_of(self, y_of_lnm, ell_max=3):
        """Plain central moments of y(lnM) under each bin's weight.

        Returns (norm, ybar, mu) with mu[b, ell] for 0 <= ell <= ell_max;
        mu[:, 0] == 1 and mu[:, 1] == 0 by construction.
        """
        y = np.asarray(y_of_lnm(self.lnm_x), dtype=float)
        wW = self.W * self.lnm_w[None, :]
        norm = wW.sum(axis=1)
        ybar = (wW @ y) / norm
        d = y[None, :] - ybar[:, None]
        mu = np.empty((self.n_bins, ell_max + 1))
        for ell in range(ell_max + 1):
            mu[:, ell] = np.sum(wW * d**ell, axis=1) / norm
        return norm, ybar, mu
