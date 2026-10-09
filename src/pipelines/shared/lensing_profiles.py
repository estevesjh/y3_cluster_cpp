"""Production-exact replicas of the miscentred 1-halo lensing profile.

The Shear1hMisSel mixture (lensing_weights.hh):

    DSigma_cl(R | M; bin) = (1 - f_mis) DSigma_nfw(R, M)
                          + f_mis      DSigma_mis(R, tau_mis R_lambda, M)

with the centred term interpolated from the per-sample
``haloModel/dSigma_nfw`` table (GSL bilinear, clamped) and the
miscentred term from the fixed gamma-kernel look-up table through
NFW_DSIGMA_MIS (bilinear in ln u over (ln x, ln x_mis), clamped, with
the analytic 2 r_s delta_c rho_ref amplitude and the 1e-12
Mpc^2 -> pc^2 conversion).

These are interpolation-exact replicas — linear interpolation with
clamped queries is the same arithmetic GSL performs — so Python
implementations that consume them can match the production .so to
near machine precision. (Contrast with
observables/shear_1h2h/python/0d/nfw_profile_family.py,
which deliberately uses a *smooth* spline view of the same gamma table
for offline differentiation.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

CONC = 4.0
RHOC = 2.77533742639e11
DELTA_C = (200.0 * CONC**3 / 3.0) / (np.log(1.0 + CONC) - CONC / (1.0 + CONC))

GAMMA_TABLE = "table_1000_1e-03_5e+03_{}_{}.txt"


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "data" / "nfw_off_center").is_dir():
            return p
    raise FileNotFoundError("lensing_profiles: cannot locate data/")


def r_lambda(lob):
    """R_lambda(lambda^ob) = (lambda^ob / 100)^0.2 [Mpc/h]."""
    return (np.asarray(lob, dtype=float) / 100.0) ** 0.2


class HaloModelDSigmaNfw:
    """Bilinear clamp interpolator over haloModel/dSigma_nfw(r_sigma, lnM)."""

    def __init__(self, source):
        self._r = source.array("halomodel", "r_sigma")
        self._lnm = source.array("halomodel", "lnm")
        vals = np.asarray(source.array("halomodel", "dsigma_nfw"), dtype=float)
        if vals.shape != (self._lnm.size, self._r.size):
            vals = vals.reshape(self._lnm.size, self._r.size)
        self._interp = RegularGridInterpolator(
            (self._lnm, self._r), vals, method="linear",
            bounds_error=False, fill_value=None)

    def __call__(self, r_perp, lnM):
        r = np.clip(np.asarray(r_perp, dtype=float), self._r[0], self._r[-1])
        m = np.clip(np.asarray(lnM, dtype=float), self._lnm[0], self._lnm[-1])
        m_b, r_b = np.broadcast_arrays(m, r)
        return self._interp(np.stack([m_b, r_b], axis=-1))


class NfwDsigmaMisProduction:
    """NFW_DSIGMA_MIS replica, clamped bilinear.

    Mirrors nfw_dsigma_mis.hh's kernel_is_signed contract: the "single"
    kernel reads the SIGNED (linear) table table_ratio_deltasigma_signed_
    single.txt, gridded on the CUSP-SAFE (ln x_mis, ln q) axes (q = x/x_mis
    -- imported from CLensPy, see
    data/nfw_off_center/import_clenspy_single_table.py and GitHub issue #6)
    and must NOT be exponentiated -- the negative lobe at r_mis > r is
    physical and is what makes the mean-field ("rnd") projection term
    cancel. The legacy "gamma" kernel stays on the natural (ln x_mis, ln x)
    grid with the log table + exp() read (no CLensPy equivalent yet).
    """

    def __init__(self, kernel="gamma", data_dir=None, rho_ref=None):
        """rho_ref: the UNIFIED reference density (haloModel/rho_m_ref =
        Omega_m rho_crit,0 (1+z_density)^3) driving BOTH the halo boundary
        r_200 = [3M/(800 pi rho_ref)]^(1/3) and the amplitude
        rho_s = delta_c rho_ref -- mirrors NFW_DSIGMA_MIS::set_rho_ref
        (2026-08-24 convention decision). Default None keeps the legacy
        rho_crit class-level convention for dump-free unit tests."""
        d = Path(data_dir) if data_dir else _repo_root() / "data" / "nfw_off_center"
        self.rho_ref = RHOC if rho_ref is None else float(rho_ref)
        self._is_signed = kernel == "single"
        self._lnxm = np.loadtxt(d / GAMMA_TABLE.format(kernel, "logxmis")
                                 if not self._is_signed
                                 else d / "table_ratio_logxmis.txt")
        if self._is_signed:
            self._lnq = np.loadtxt(d / "table_ratio_logq.txt")
            tab = np.loadtxt(d / "table_ratio_deltasigma_signed_single.txt")
            self._interp = RegularGridInterpolator(
                (self._lnxm, self._lnq), tab, method="linear",
                bounds_error=False, fill_value=None)
        else:
            self._lnx = np.loadtxt(d / GAMMA_TABLE.format(kernel, "logx"))
            tab = np.loadtxt(
                d / f"table_1000_1e-03_5e+03_log_deltasigma_{kernel}.txt")
            self._interp = RegularGridInterpolator(
                (self._lnxm, self._lnx), tab, method="linear",
                bounds_error=False, fill_value=None)

    def __call__(self, r_perp, r_mis, lnM, rho_mult=1.0, conc=None):
        """conc: optional per-mass concentration, broadcastable with lnM
        (issue #13; mirrors NFW_DSIGMA_MIS::set_concentration_table).
        The lookup table is universal in x = r/r_s, so c enters only the
        analytic r_s = r_200/c and delta_c(c). Default None = legacy
        fixed c = CONC (= 4). rho_mult is a PURE EXTERNAL amplitude
        factor (it never touches the boundary); production paths pass
        rho_ref at construction instead."""
        lnM = np.asarray(lnM, dtype=float)
        if conc is None:
            c, delta_c = CONC, DELTA_C
        else:
            c = np.asarray(conc, dtype=float)
            delta_c = (200.0 * c**3 / 3.0) / (np.log1p(c) - c / (1.0 + c))
        r_200 = np.cbrt(3.0 * np.exp(lnM) / (800.0 * np.pi * self.rho_ref))
        r_s = r_200 / c
        x = np.asarray(r_perp, dtype=float) / r_s
        xmis = np.asarray(r_mis, dtype=float) / r_s
        lnxm = np.clip(np.log(xmis), self._lnxm[0], self._lnxm[-1])
        if self._is_signed:
            lnq = np.clip(np.log(x / xmis), self._lnq[0], self._lnq[-1])
            lnxm_b, lnq_b = np.broadcast_arrays(lnxm, lnq)
            u = self._interp(np.stack([lnxm_b, lnq_b], axis=-1))
        else:
            lnx = np.clip(np.log(x), self._lnx[0], self._lnx[-1])
            lnxm_b, lnx_b = np.broadcast_arrays(lnxm, lnx)
            u = self._interp(np.stack([lnxm_b, lnx_b], axis=-1))
        # signed kernels tabulate DeltaSigma directly; legacy log kernels
        # tabulate log(DeltaSigma) (see class docstring).
        val = u if self._is_signed else np.exp(u)
        norm = 2.0 * r_s * delta_c * self.rho_ref * rho_mult
        return norm * val * 1.0e-12


class MisMixtureProfile:
    """The per-bin production mixture Phi_i(R, lnM) with per-sample knobs."""

    def __init__(self, source, *, lob_centers, f_mis, tau_mis, omega_m):
        self._cen = HaloModelDSigmaNfw(source)
        # UNIFIED rho_m convention: the mis component shares the density
        # the centred dSigma_nfw table was built with (haloModel/rho_m_ref).
        self.rho_ref = float(source.scalar("halomodel", "rho_m_ref"))
        self._mis = NfwDsigmaMisProduction(
            kernel="gamma", rho_ref=self.rho_ref)
        self._lob = np.asarray(lob_centers, dtype=float)
        self.f_mis = float(f_mis)
        self.tau_mis = float(tau_mis)
        self.omega_m = float(omega_m)

    def r_mis(self, bin_index):
        """tau_mis R_lambda for richness bin = bin_index mod len(lob)."""
        return self.tau_mis * float(r_lambda(self._lob[bin_index % self._lob.size]))

    def __call__(self, bin_index, r_perp, lnM, q=1.0):
        """q: physical-density query rescale (1 + z), applied to BOTH
        comoving query radii -- the radius half of the exact identity
        DSigma_phys(R|z) = (1+z)^2 DSigma_com(R (1+z)). The (1+z)^2
        amplitude is the CALLER's (z-weight or explicit factor), matching
        the C++ evaluators."""
        d_cen = self._cen(np.asarray(r_perp) * q, lnM)
        d_mis = self._mis(np.asarray(r_perp) * q, self.r_mis(bin_index) * q,
                          lnM)
        return (1.0 - self.f_mis) * d_cen + self.f_mis * d_mis


class MaxMixtureProfile:
    """Traditional 1h+2h shear profile: the SIG_MAX/GAMMA_MAX composition
    with the modern haloModel tables,

        DSigma_max(b, R, lnM, z) = max( DSigma_cl(R, lnM | bin b),
                                        bias(lnM, z) * dSigma_hh(R, z) )

    where DSigma_cl is the production miscentred 1-halo mixture
    (MisMixtureProfile; set include_miscentering=False for the pure
    centred term) and the two-halo term is z-dependent — callers must
    keep z inside the mass integral (see
    explicit_grid_core.explicit_mass_z_weights).
    """

    def __init__(self, source, *, lob_centers, f_mis, tau_mis, omega_m,
                 include_miscentering=True):
        if include_miscentering:
            self._one = MisMixtureProfile(source, lob_centers=lob_centers,
                                          f_mis=f_mis, tau_mis=tau_mis,
                                          omega_m=omega_m)
        else:
            self._one = MisMixtureProfile(source, lob_centers=lob_centers,
                                          f_mis=0.0, tau_mis=tau_mis,
                                          omega_m=omega_m)
        # local import to avoid a cycle at module import time
        from . import datablock_models as _dm
        self._bias = _dm.Bilinear2D(source, "halomodel", "lnm", "z", "bias")
        # The Hankel-based producer leaves dSigma_hh undefined (NaN) at
        # low radii (plan owner: expected); the max model resolves to
        # the 1-halo term there, so NaN -> 0 before interpolation is
        # the faithful treatment (b*0 never wins the max where 1h is
        # finite).
        self._hh = _dm.Bilinear2D(source, "halomodel", "r_sigma", "z",
                                  "dsigma_hh", nan_fill=0.0)

    def __call__(self, b, r_perp, lnM, z):
        DSigma_1h = self._one(b, r_perp, lnM)
        DSigma_2h = self._bias(lnM, z) * self._hh(r_perp, z)
        return np.maximum(DSigma_1h, DSigma_2h)


class Shear1h2hMaxProfile:
    r"""The centred 3D profile used by the ``Shear1h2hMax`` model.

    The model is defined in real space,

    ``xi_hm(r) = max(xi_1h(r), bias(lnM, z) * xi_nl(r, z))``,

    and projected into explicit lensing observables:

    ``Sigma(R)`` is the projected surface density,
    ``mean_surface_density(R)`` is ``barSigma(<R)``,
    ``excess_surface_density(R)`` is ``DeltaSigma(R) = barSigma - Sigma``,
    and ``tangential_shear(R)`` is
    ``gamma_T(R) = DeltaSigma(R) * sigma_crit_inv``.

    All returned surface densities use the same comoving mass/area units as
    the published ``haloModel`` tables.  ``tangential_shear`` requires the
    caller to provide ``sigma_crit_inv`` in the reciprocal matching units;
    geometry is never hidden inside this profile.

    The profile is evaluated with the three-pillar construction documented in
    ``plans/Fast and Precise Numerical Framework for 3D Halo Exclusion
    Weak Lensing.md``.  The outer branch uses the exact ``C0 / R**2``
    decoupled tail; the inner branch uses nonsingular GL integrals after the
    ``y = R sinh(u)`` substitution.  In
    ``excess_surface_density_grid``, the NFW gamma table is retained as the
    baseline and the signed residual
    ``Sigma_max_cen - Sigma_NFW_cen`` is convolved separately.  This is the
    additive correction implied by linearity; no NFW response ratio is
    transferred to the complete max profile.

    This class deliberately consumes the published ``xi_nl`` and cumulative
    ``m3d_nl`` tables.  It does not recompute the nonlinear correlation
    function or perform an adaptive integral in the observable loop.
    """

    def __init__(self, source, *, lob_centers, f_mis, tau_mis, omega_m,
                 include_miscentering=True, n_gl=20, n_root=96,
                 n_offset=48, n_phi=128, n_aperture=8, q_max=24.0,
                 n_residual=512, use_nfw_table_residual=True):
        from . import datablock_models as _dm

        self._cen = HaloModelDSigmaNfw(source)
        self._lnm = source.array("halomodel", "lnm")
        self._rho_ref = float(source.scalar("halomodel", "rho_m_ref"))
        self._omega_m = float(omega_m)
        self._rho_m = self._omega_m * RHOC
        self._bias = _dm.Bilinear2D(source, "halomodel", "lnm", "z",
                                    "bias")
        self._xi = _dm.Bilinear2D(source, "xi_nl", "r", "z", "xi_nl",
                                  nan_fill=0.0)
        self._m3d = _dm.Bilinear2D(source, "xi_nl", "r", "z", "m3d_nl",
                                   nan_fill=0.0)
        self._sigma_bar_hh = _dm.Bilinear2D(
            source, "halomodel", "r_sigma", "z", "sigma_bar_hh",
            nan_fill=0.0)
        self._dsigma_hh = _dm.Bilinear2D(
            source, "halomodel", "r_sigma", "z", "dsigma_hh",
            nan_fill=0.0)
        self._r_xi = source.array("xi_nl", "r")
        self._r_sigma = source.array("halomodel", "r_sigma")
        self._c = (np.asarray(source.array("halomodel", "concentration"),
                              dtype=float)
                   if source.has("halomodel", "concentration")
                   else np.full(self._lnm.size, CONC))
        self._conc = np.interp(self._lnm, self._lnm, self._c)
        self._mis = NfwDsigmaMisProduction(
            kernel="gamma", rho_ref=self._rho_ref)
        self._lob = np.asarray(lob_centers, dtype=float)
        self.f_mis = float(f_mis) if include_miscentering else 0.0
        self.tau_mis = float(tau_mis)
        self.n_gl = int(n_gl)
        self.n_root = int(n_root)
        self.n_offset = int(n_offset)
        self.n_phi = int(n_phi)
        self.n_aperture = int(n_aperture)
        self.q_max = float(q_max)
        self.n_residual = int(n_residual)
        self.use_nfw_table_residual = bool(use_nfw_table_residual)
        if self.n_gl < 1:
            raise ValueError("n_gl must be positive")
        if self.n_root < 2:
            raise ValueError("n_root must be at least 2")
        if self.n_offset < 1 or self.n_phi < 1 or self.n_aperture < 1:
            raise ValueError("miscentering quadrature orders must be positive")
        if self.q_max <= 0.0 or self.n_residual < 16:
            raise ValueError("invalid residual-convolution grid configuration")
        self._gl_x, self._gl_w = np.polynomial.legendre.leggauss(self.n_gl)
        self._phi_t, self._phi_w = self._clenspy_gl_nodes(
            0.0, 1.0, self.n_phi)
        self._aperture_t, self._aperture_w = self._clenspy_gl_nodes(
            0.0, 1.0, self.n_aperture)
        self._transition_cache = {}

    @staticmethod
    def _clenspy_gl_nodes(a, b, n):
        """Use CLensPy's fixed Gauss--Legendre node generator."""
        from clenspy.utils.integrate import gl_nodes
        return gl_nodes(a, b, n)

    def r_mis(self, bin_index):
        return self.tau_mis * float(r_lambda(
            self._lob[bin_index % self._lob.size]))

    def _concentration(self, lnM):
        return float(np.interp(lnM, self._lnm, self._c))

    def _nfw_parameters(self, lnM):
        c = self._concentration(lnM)
        r200 = np.cbrt(3.0 * np.exp(lnM) /
                       (800.0 * np.pi * self._rho_ref))
        rs = r200 / c
        delta_c = (200.0 * c**3 / 3.0) / (np.log1p(c) - c / (1.0 + c))
        rho_s = delta_c * self._rho_ref
        return rs, rho_s, r200

    @staticmethod
    def _nfw_shapes(x):
        """Return the standard projected NFW shapes ``f(x)`` and ``g(x)``."""
        x = np.maximum(np.asarray(x, dtype=float), 1.0e-12)
        f = np.empty_like(x)
        g = np.empty_like(x)
        below = x < 1.0 - 1.0e-8
        above = x > 1.0 + 1.0e-8
        at_one = ~(below | above)
        xb = x[below]
        xa = x[above]
        root_b = np.sqrt((1.0 - xb) / (1.0 + xb))
        f[below] = (
            1.0 - 2.0 * np.arctanh(root_b) / np.sqrt(1.0 - xb * xb)
        ) / (xb * xb - 1.0)
        g[below] = (
            np.log(xb / 2.0)
            + 2.0 * np.arctanh(root_b) / np.sqrt(1.0 - xb * xb)
        )
        root_a = np.sqrt((xa - 1.0) / (1.0 + xa))
        f[above] = (
            1.0 - 2.0 * np.arctan(root_a) / np.sqrt(xa * xa - 1.0)
        ) / (xa * xa - 1.0)
        g[above] = (
            np.log(xa / 2.0)
            + 2.0 * np.arctan(root_a) / np.sqrt(xa * xa - 1.0)
        )
        f[at_one] = 1.0 / 3.0
        g[at_one] = 1.0 + np.log(0.5)
        return f, g

    def _nfw_surface_density(self, r_perp, lnM):
        rs, rho_s, _ = self._nfw_parameters(lnM)
        x = np.asarray(r_perp, dtype=float) / rs
        f, _ = self._nfw_shapes(x)
        value = 2.0 * rho_s * rs * f * 1.0e-12
        return float(value) if np.ndim(value) == 0 else value

    def _nfw_excess_surface_density(self, r_perp, lnM):
        rs, rho_s, _ = self._nfw_parameters(lnM)
        x = np.asarray(r_perp, dtype=float) / rs
        f, g = self._nfw_shapes(x)
        value = 2.0 * rho_s * rs * (2.0 * g / x**2 - f) * 1.0e-12
        return float(value) if np.ndim(value) == 0 else value

    def _residual_surface_density(self, r_perp, lnM, z):
        """Signed ``Sigma_max_cen - Sigma_NFW_cen``."""
        return (self.surface_density(r_perp, lnM, z)
                - self._nfw_surface_density(r_perp, lnM))

    def _residual_interpolator(self, r_max, lnM, z):
        """Build a signed log-radius interpolator for the residual Sigma."""
        r_min = 1.0e-8
        r_max = max(float(r_max), 10.0 * r_min)
        radii = np.geomspace(r_min, r_max, self.n_residual)
        values = np.array([
            self._residual_surface_density(radius, lnM, z)
            for radius in radii
        ])
        return np.log(radii), PchipInterpolator(
            np.log(radii), values, extrapolate=False)

    def _gamma_residual_surface_density(self, radius, r_mis, log_r, interp):
        """Gamma-convolve the signed residual surface density."""
        radius = max(float(radius), 1.0e-8)
        if r_mis <= 0.0:
            return float(interp(np.clip(np.log(radius), log_r[0], log_r[-1])))
        edges = [0.0, 1.0, 4.0, 8.0, 16.0, self.q_max]
        q_break = radius / r_mis
        if 0.0 < q_break < self.q_max:
            edges.append(q_break)
        edges = np.unique(np.clip(edges, 0.0, self.q_max))
        total = 0.0
        phi = np.pi * self._phi_t**2
        w_phi = 2.0 * self._phi_t * self._phi_w
        for left, right in zip(edges[:-1], edges[1:]):
            if right <= left:
                continue
            q, w_q = self._clenspy_gl_nodes(left, right, self.n_offset)
            offset = r_mis * q
            separation = np.sqrt(
                (radius - offset[:, None])**2
                + 4.0 * radius * offset[:, None]
                * np.sin(phi[None, :] / 2.0)**2
            )
            log_sep = np.clip(
                np.log(np.maximum(separation, 1.0e-8)),
                log_r[0], log_r[-1])
            angular = np.sum(interp(log_sep) * w_phi[None, :], axis=1)
            total += np.sum(w_q * q * np.exp(-q) * angular)
        return float(total)

    def _gamma_residual_excess_surface_density(
            self, radius, r_mis, log_r, interp):
        """Aperture-project the gamma-convolved residual Sigma."""
        radius = max(float(radius), 1.0e-8)
        x = radius * self._aperture_t**2
        values = np.array([
            self._gamma_residual_surface_density(value, r_mis, log_r, interp)
            for value in x
        ])
        mean = 4.0 * np.sum(
            self._aperture_w * self._aperture_t**3 * values)
        edge = self._gamma_residual_surface_density(
            radius, r_mis, log_r, interp)
        return float(mean - edge)

    def _xi_1h(self, r, lnM):
        rs, rho_s, _ = self._nfw_parameters(lnM)
        x = np.maximum(np.asarray(r, dtype=float) / rs, 1.0e-15)
        return rho_s / (self._rho_ref * x * (1.0 + x)**2) - 1.0

    def _m3d_1h(self, r, lnM):
        rs, rho_s, _ = self._nfw_parameters(lnM)
        r = np.asarray(r, dtype=float)
        x = r / rs
        return (4.0 * np.pi * rho_s * rs**3
                * (np.log1p(x) - x / (1.0 + x))
                - (4.0 / 3.0) * np.pi * self._rho_ref * r**3)

    def _xi_2h(self, r, lnM, z):
        return self._bias(lnM, z) * self._xi(r, z)

    def _m3d_2h(self, r, lnM, z):
        return self._bias(lnM, z) * self._m3d(r, z)

    def _transition_radius(self, lnM, z):
        key = (float(lnM), float(z))
        if key in self._transition_cache:
            return self._transition_cache[key]
        _, _, r200 = self._nfw_parameters(lnM)
        lo = max(float(self._r_xi[0]), 0.25 * r200)
        hi = float(self._r_xi[-1])
        trial = np.geomspace(lo, hi, self.n_root)
        diff = self._xi_1h(trial, lnM) - self._xi_2h(trial, lnM, z)
        valid = np.isfinite(diff)
        crossing = None
        for i in range(trial.size - 1):
            if valid[i] and valid[i + 1] and diff[i] >= 0.0 and diff[i + 1] <= 0.0:
                crossing = (trial[i], trial[i + 1])
                break
        if crossing is None:
            # The published tables are expected to contain a crossing.  A
            # clamped endpoint keeps diagnostic runs finite for synthetic or
            # deliberately truncated inputs while preserving the correct
            # branch whenever a crossing is available.
            out = float(np.clip(2.0 * r200, self._r_xi[0], self._r_xi[-1]))
            self._transition_cache[key] = out
            return out
        a, b = crossing
        fa = float(np.asarray(
            self._xi_1h(a, lnM) - self._xi_2h(a, lnM, z)).item())
        for _ in range(60):
            mid = np.sqrt(a * b)
            fm = float(np.asarray(
                self._xi_1h(mid, lnM) - self._xi_2h(mid, lnM, z)).item())
            if fa * fm > 0.0:
                a, fa = mid, fm
            else:
                b = mid
        out = np.sqrt(a * b)
        self._transition_cache[key] = float(out)
        return float(out)

    def _projection_corrections(self, R, lnM, z):
        """Return ``(Sigma, barSigma, DeltaSigma)`` exclusion corrections."""
        R = max(float(R), self._r_sigma[0])
        r_trans = self._transition_radius(lnM, z)
        delta_m_trans = (float(self._m3d_1h(r_trans, lnM))
                         - float(np.asarray(
                             self._m3d_2h(r_trans, lnM, z)).item()))
        outer_tail = delta_m_trans / (np.pi * R * R) * 1.0e-12
        if R >= r_trans:
            return 0.0, outer_tail, outer_tail

        # y = R sinh(u), r = R cosh(u), dy = r du.  This resolves the
        # narrow NFW core while keeping both projected integrands smooth.
        u_max = np.arccosh(r_trans / R)
        u = 0.5 * u_max * (self._gl_x + 1.0)
        wu = 0.5 * u_max * self._gl_w
        y = R * np.sinh(u)
        rr = R * np.cosh(u)
        delta_xi = self._xi_1h(rr, lnM) - self._xi_2h(rr, lnM, z)
        sigma_diff = 2.0 * self._rho_ref * np.sum(wu * rr * delta_xi)
        mantle = 4.0 * self._rho_ref * np.sum(
            wu * rr * delta_xi * y / (rr + y))
        delta_m = (float(self._m3d_1h(R, lnM))
                   - float(np.asarray(self._m3d_2h(R, lnM, z)).item()))
        mean_diff = delta_m / (np.pi * R * R) + mantle
        return (sigma_diff * 1.0e-12, mean_diff * 1.0e-12,
                (mean_diff - sigma_diff) * 1.0e-12)

    def surface_density(self, r_perp, lnM, z):
        r"""Return projected surface density ``Sigma(R)``."""
        R = max(float(r_perp), self._r_sigma[0])
        bias = float(np.asarray(self._bias(lnM, z)).item())
        sigma_2h = bias * (
            float(np.asarray(self._sigma_bar_hh(R, z)).item())
            - float(np.asarray(self._dsigma_hh(R, z)).item()))
        return sigma_2h + self._projection_corrections(R, lnM, z)[0]

    def mean_surface_density(self, r_perp, lnM, z):
        r"""Return mean interior surface density ``barSigma(<R)``."""
        R = max(float(r_perp), self._r_sigma[0])
        bias = float(np.asarray(self._bias(lnM, z)).item())
        mean_2h = bias * float(np.asarray(
            self._sigma_bar_hh(R, z)).item())
        return mean_2h + self._projection_corrections(R, lnM, z)[1]

    def excess_surface_density(self, r_perp, lnM, z):
        r"""Return excess surface density ``DeltaSigma = barSigma - Sigma``."""
        R = max(float(r_perp), self._r_sigma[0])
        bias = float(np.asarray(self._bias(lnM, z)).item())
        dsigma_2h = bias * float(np.asarray(self._dsigma_hh(R, z)).item())
        return dsigma_2h + self._projection_corrections(R, lnM, z)[2]

    def tangential_shear(self, r_perp, lnM, z, sigma_crit_inv):
        r"""Return ``gamma_T = DeltaSigma * sigma_crit_inv``."""
        return (self.excess_surface_density(r_perp, lnM, z)
                * sigma_crit_inv)

    def excess_surface_density_grid(self, bin_index, r_perp, lnM, z,
                                    physical=False):
        """Return the miscentred ``DeltaSigma(R, lnM, z)`` grid.

        The NFW table supplies the gamma-convolved NFW component.  The
        complete max-minus-NFW residual is convolved separately at Sigma
        level, then aperture projected.  A second ``max`` after
        miscentering is not equivalent and can remove the signed inner lobe.
        """
        r_perp = np.asarray(r_perp, dtype=float)
        lnM = np.asarray(lnM, dtype=float)
        z = np.asarray(z, dtype=float)
        out = np.empty((r_perp.size, lnM.size, z.size))
        r_mis_base = self.r_mis(bin_index)
        for ik, mass in enumerate(lnM):
            for iz, redshift in enumerate(z):
                q = 1.0 + redshift if physical else 1.0
                log_r = residual = None
                if self.use_nfw_table_residual and self.f_mis != 0.0:
                    log_r, residual = self._residual_interpolator(
                        np.max(r_perp * q) + self.q_max * r_mis_base * q,
                        mass, redshift)
                for ir, R in enumerate(r_perp):
                    q = 1.0 + redshift if physical else 1.0
                    centered_com = self.excess_surface_density(
                        R * q, mass, redshift)
                    centered = centered_com * q**2
                    d_cen = float(np.asarray(
                        self._cen(R * q, mass)).item()) * q**2
                    mis = float(np.asarray(
                        self._mis(R * q, r_mis_base * q, mass)).item()) * q**2
                    if not self.use_nfw_table_residual or self.f_mis == 0.0:
                        out[ir, ik, iz] = (
                            centered + self.f_mis * (mis - d_cen))
                        continue
                    nfw_cen = self._nfw_excess_surface_density(R * q, mass)
                    d_res_cen = (centered_com - nfw_cen) * q**2
                    d_res_mis = self._gamma_residual_excess_surface_density(
                        R * q, r_mis_base * q, log_r, residual) * q**2
                    out[ir, ik, iz] = (
                        centered + self.f_mis * (
                            mis + d_res_mis - nfw_cen * q**2
                            - d_res_cen))
        return out
