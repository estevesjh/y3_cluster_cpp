"""Gamma-kernel miscentering of an excess surface density, fixed Gauss-Legendre.

Python twin of the gauss_legendre path of ``Miscentering.hh`` (Miscentering,
MiscenteringOperator, LogRadialGrid: same nodes, same equations).

For a halo with centred surface density Sigma(r), a fraction f_mis of clusters
is centred at a projected offset R_s ~ P(R_s | sigma) = R_s/sigma^2 exp(-R_s/sigma),
sigma_b = tau_mis R_lambda(lambda_b):

  (E1) Sigma_mis(R)  = int dR_s P(R_s) (1/pi) int_0^pi dtheta
                         Sigma(sqrt(R^2 + R_s^2 + 2 R R_s cos theta))
  (E2) DSigma_mis(R) = int_0^1 4 t^3 Sigma_mis(R t^2) dt - Sigma_mis(R)
  (E3) O_obs         = (1 - f_mis) O_cen + f_mis O_mis

GL nodes: offsets on [0, R] and [R, R + x_max sigma]; theta = pi (1 - v^2);
aperture R' = R t^2.  On a log-uniform grid with Sigma linear in ln r,
(E1)+(E2) is one weight row per (sigma, R): DSigma_mis(R) = sum_m L_m Sigma(r_m).
Ring radii reach 2 R + x_max sigma.  Below the grid floor the profile must keep
its core mass (clamping below 0.1 Mpc/h biases DSigma_mis by 23% at R = 0.2):
start the grid inside the cusp, or use a core slot holding the uniform disk
M_2d(<r_lo) / (pi r_lo^2) (haloModel/m2d_core; 1.4e-5 at r_lo = 1e-3).

The adaptive Cuhre reference (mis_method = cuhre) and the analytic NFW
validation profile are C++ only (Miscentering.hh).
"""
from __future__ import annotations

import numpy as np

from shared import datablock_models as dm

SECTION = "miscentering"


class LogRadialGrid:
    """r_m = exp(ln r_lo + m dln), hat functions linear in ln r, clamped above;
    below r_lo clamped to r_0, or with core=True sent to the core slot n."""

    def __init__(self, r_lo, r_hi, n, core=False):
        if not (r_lo > 0.0 and r_hi > r_lo and n >= 2):
            raise ValueError("LogRadialGrid: needs 0 < r_lo < r_hi and n >= 2")
        self.core = bool(core)
        self.ln_lo = np.log(r_lo)
        self.dln = (np.log(r_hi) - self.ln_lo) / (n - 1)
        self.r = np.exp(self.ln_lo + self.dln * np.arange(n))

    @property
    def r_hi(self):
        return self.r[-1]

    @property
    def row_size(self):
        return self.r.size + (1 if self.core else 0)

    def scatter(self, ln_r, w, row):
        """row += w h_m(r) (vectorised over ln_r, w)."""
        last = self.r.size - 1
        u = (np.ravel(ln_r) - self.ln_lo) / self.dln
        w = np.ravel(w)
        if self.core:
            inside = u < 0.0
            row[last + 1] += np.sum(w[inside])
            u, w = u[~inside], w[~inside]
        i = np.clip(np.floor(u).astype(int), 0, last - 1)
        f = np.clip(u - i, 0.0, 1.0)
        np.add.at(row, i, w * (1.0 - f))
        np.add.at(row, i + 1, w * f)


class MiscenteringOperator:
    """Fixed-GL gamma miscentering; mirrors y3_cluster::MiscenteringOperator."""

    DEFAULT_N = 32
    DEFAULT_X_MAX = 20.0

    def __init__(self, f_mis, tau_mis, n_offset=DEFAULT_N, n_theta=DEFAULT_N,
                 n_aperture=DEFAULT_N, x_max=DEFAULT_X_MAX):
        if not 0.0 <= f_mis <= 1.0:
            raise ValueError("MiscenteringOperator: f_mis must be in [0, 1]")
        if not (np.isfinite(tau_mis) and tau_mis > 0.0):
            raise ValueError("MiscenteringOperator: tau_mis must be > 0")
        if min(n_offset, n_theta, n_aperture) < 1 or not x_max > 0.0:
            raise ValueError("MiscenteringOperator: needs n >= 1 and x_max > 0")
        self.f_mis, self.tau_mis, self.x_max = float(f_mis), float(tau_mis), float(x_max)
        self._u, self._wu = dm.gl_nodes(0.0, 1.0, n_offset)
        v, wv = dm.gl_nodes(0.0, 1.0, n_theta)
        self._cos_theta = np.cos(np.pi * (1.0 - v * v))
        self._w_theta = 2.0 * v * wv
        t, wt = dm.gl_nodes(0.0, 1.0, n_aperture)
        self._t2 = t * t
        self._w_ap = 4.0 * wt * t**3

    @classmethod
    def from_source(cls, source, section=SECTION, **sizes):
        """Required: miscentering/f_mis and tau_mis (no fiducial fallback)."""
        return cls(source.scalar(section, "f_mis"), source.scalar(section, "tau_mis"), **sizes)

    def offset_scale(self, lob):
        return self.tau_mis * dm.R_lambda(lob)

    def max_ring_radius(self, R, sigma):
        return 2.0 * R + self.x_max * sigma

    @staticmethod
    def _gamma_pdf(rs, sigma):
        return rs / sigma**2 * np.exp(-rs / sigma)

    def _ring_nodes(self, Rp, sigma):
        """(D1) nodes for Sigma_mis at each Rp: (r^2, weight), shape (n_Rp, 2 n_off * n_th)."""
        Rp = np.atleast_1d(np.asarray(Rp, float))[:, None]
        if np.any(Rp <= 0.0) or not sigma > 0.0:
            raise ValueError("MiscenteringOperator requires R > 0, sigma > 0")
        tail = self.x_max * sigma
        rs = np.concatenate([Rp * self._u, Rp + tail * self._u], axis=1)
        w = np.concatenate([Rp * self._wu * self._gamma_pdf(Rp * self._u, sigma),
                            tail * self._wu * self._gamma_pdf(Rp + tail * self._u, sigma)], axis=1)
        r2 = (Rp[:, :, None]**2 + rs[:, :, None]**2
              + 2.0 * Rp[:, :, None] * rs[:, :, None] * self._cos_theta)
        return r2.reshape(Rp.shape[0], -1), (w[:, :, None] * self._w_theta).reshape(Rp.shape[0], -1)

    def _aperture_radii(self, R):
        return np.concatenate([[R], R * self._t2]), np.concatenate([[-1.0], self._w_ap])

    def excess_surface_density_row(self, R, sigma, grid):
        """(D4): DSigma_mis(R) = row @ Sigma(grid.r)."""
        if self.max_ring_radius(R, sigma) > grid.r_hi * (1.0 + 1.0e-12):
            raise ValueError("MiscenteringOperator: radial grid ends below 2 R + x_max sigma")
        radii, coef = self._aperture_radii(R)
        r2, w = self._ring_nodes(radii, sigma)
        row = np.zeros(grid.row_size)
        grid.scatter(0.5 * np.log(r2), coef[:, None] * w, row)
        return row

    def apply(self, o_cen, o_mis):
        """(E4)"""
        return (1.0 - self.f_mis) * o_cen + self.f_mis * o_mis


class Miscentering:
    """Stacked gamma miscentering for a z-resolved (lnM_k, z_q) consumer;
    mirrors y3_cluster::Miscentering (gauss_legendre only — the Cuhre
    reference is C++).

        O_mis(b, R) = sum_{k,q} W(b; k, q) DSigma_mis(R | k, q; sigma_b),
        O           = (1 - f_mis) O_cen + f_mis O_mis               (apply)

    of the consumer's centred profile B_sel Sigma.  Sigma(r | k, q) is
    tabulated once per sample on a LogRadialGrid (core slot when the
    profile has a core), MiscenteringOperator rows per (richness slot, wall
    R) are cached while tau_mis and the grid floor are unchanged, and the
    stack is folded in once: g_b[m] = sum_q B(r_m; z_q) sum_k W Sigma, so
    evaluate() is a dot product.
    """

    METHODS = ("gauss_legendre", "cuhre")

    def __init__(self, wall_radii, lob_centers, method="gauss_legendre",
                 n_offset=32, n_theta=32, n_aperture=32, n_r=1024):
        if method == "cuhre":
            raise NotImplementedError(
                "Miscentering (Python): mis_method = cuhre is C++ only; "
                "use the C++ module for the Cuhre reference")
        if method not in self.METHODS:
            raise ValueError(f"Miscentering: unknown mis_method {method}")
        self.wall = np.asarray(wall_radii, dtype=float)
        self.lob = np.asarray(lob_centers, dtype=float)
        if self.wall.size == 0 or self.lob.size == 0:
            raise ValueError("Miscentering: needs wall radii and lob_centers")
        self.method = method
        self.n_offset, self.n_theta, self.n_aperture, self.n_r = (
            int(n_offset), int(n_theta), int(n_aperture), int(n_r))
        self._key = None
        self.op = None

    @classmethod
    def from_options(cls, options, section, wall_radii, lob_centers):
        """Module-section keys: mis_method, n_mis_offset, n_mis_theta,
        n_mis_aperture, n_r_mis."""
        def opt(get, key, default):
            return get(section, key) if options.has_value(section, key) else default
        return cls(wall_radii, lob_centers,
                   method=opt(options.get_string, "mis_method", "gauss_legendre"),
                   n_offset=opt(options.get_int, "n_mis_offset", 32),
                   n_theta=opt(options.get_int, "n_mis_theta", 32),
                   n_aperture=opt(options.get_int, "n_mis_aperture", 32),
                   n_r=opt(options.get_int, "n_r_mis", 1024))

    def set_sample(self, source, *, weights, z, bsel, r_lo, r_max, core, grid):
        """weights: W (b, k, q); z: z nodes; bsel: BselModels or None;
        r_lo / r_max: profile grid floor / validity; core: (k, q) uniform-disk
        Sigma below r_lo or None; grid(r) -> Sigma (m, k, q) for radii r."""
        op = MiscenteringOperator.from_source(
            source, n_offset=self.n_offset, n_theta=self.n_theta,
            n_aperture=self.n_aperture)
        self.sigma = op.offset_scale(self.lob)
        r_hi = op.max_ring_radius(np.max(self.wall), np.max(self.sigma))
        if r_hi > r_max:
            raise RuntimeError("Miscentering: the profile must reach "
                               "2 max(r_perp) + x_max sigma_max")
        if (op.tau_mis, r_lo) != self._key:
            self.grid = LogRadialGrid(r_lo, r_hi, self.n_r, core=core is not None)
            self.rows = {(j, float(R)): op.excess_surface_density_row(R, self.sigma[j], self.grid)
                         for j in range(self.lob.size) for R in self.wall}
            self._key = (op.tau_mis, r_lo)
        r = self.grid.r
        sigma = grid(r)                                                      # (m, k, q)
        r_rows = r
        if core is not None:
            sigma = np.concatenate([sigma, np.asarray(core)[None]], axis=0)
            r_rows = np.append(r, r_lo)
        gq = np.einsum("bkq,mkq->bmq", weights, sigma)
        if bsel is not None:
            for b in range(gq.shape[0]):
                gq[b] *= np.broadcast_to(bsel(
                    r_rows[:, None], bin_index=b, lob=self.lob[b % self.lob.size],
                    z=np.asarray(z)[None, :]), gq[b].shape)
        self.g = gq.sum(axis=2)                                              # (b, m)
        self.op = op

    def evaluate(self, b, R):
        """O_mis(b, R)."""
        j = b % self.lob.size
        row = self.rows.get((j, float(R)))
        if row is None:
            row = self.op.excess_surface_density_row(R, self.sigma[j], self.grid)
        return float(row @ self.g[b])

    def apply(self, o_cen, o_mis):
        """O = (1 - f_mis) O_cen + f_mis O_mis."""
        return self.op.apply(o_cen, o_mis)
