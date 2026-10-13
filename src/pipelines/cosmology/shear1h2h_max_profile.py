"""3D max(xi) halo-exclusion lensing profile.

The centred 3D profile used by the ``Shear1h2hMax`` model.
Applies ``max(xi_1h, bias * xi_nl)`` in 3D before projection.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from cosmology.nfw_model import (
    convert_m200_to_r200, f_nfw, g_nfw,
)
from shared.lensing_profiles import (
    CONC, RHOC, HaloModelDSigmaNfw, NfwDsigmaMisProduction, r_lambda,
)


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
    ``excess_surface_density_grid``, ``miscentering_method="table"`` retains
    the NFW gamma table as the baseline and convolves the signed residual
    ``Sigma_max_cen - Sigma_NFW_cen`` separately.  With
    ``miscentering_method="direct"``, the complete centered ``Sigma_max`` is
    convolved instead.  Both methods perform the gamma convolution at Sigma
    level and apply the aperture projection afterward; neither transfers an
    NFW response ratio to the complete max profile.

    This class deliberately consumes the published ``xi_nl`` and cumulative
    ``m3d_nl`` tables.  It does not recompute the nonlinear correlation
    function or perform an adaptive integral in the observable loop.
    """

    def __init__(self, source, *, lob_centers, f_mis, tau_mis, omega_m,
                 include_miscentering=True, n_gl=20, n_root=96,
                 n_offset=24, n_phi=64, n_aperture=8, q_max=12.0,
                 n_residual=512, use_nfw_table_residual=True,
                 miscentering_method="table"):
        from shared import datablock_models as _dm

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
        self.miscentering_method = str(miscentering_method).lower()
        if self.miscentering_method not in ("table", "direct"):
            raise ValueError(
                "miscentering_method must be 'table' or 'direct'")
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
        r200 = convert_m200_to_r200(np.exp(lnM), self._rho_ref)
        rs = r200 / c
        delta_c = (200.0 * c**3 / 3.0) / (np.log1p(c) - c / (1.0 + c))
        rho_s = delta_c * self._rho_ref
        return rs, rho_s, r200

    def _nfw_surface_density(self, r_perp, lnM):
        rs, rho_s, _ = self._nfw_parameters(lnM)
        x = np.atleast_1d(np.asarray(r_perp, dtype=float) / rs)
        value = 2.0 * rho_s * rs * f_nfw(x) * 1.0e-12
        return float(value.item()) if np.ndim(r_perp) == 0 else value

    def _nfw_excess_surface_density(self, r_perp, lnM):
        rs, rho_s, _ = self._nfw_parameters(lnM)
        x = np.atleast_1d(np.asarray(r_perp, dtype=float) / rs)
        value = rs * rho_s * g_nfw(x) * 1.0e-12
        return float(value.item()) if np.ndim(r_perp) == 0 else value

    def _residual_surface_density(self, r_perp, lnM, z):
        """Signed ``Sigma_max_cen - Sigma_NFW_cen``."""
        return (self.surface_density(r_perp, lnM, z)
                - self._nfw_surface_density(r_perp, lnM))

    def _surface_interpolator(self, r_max, lnM, z, residual=False):
        """Build a log-radius interpolator for centered Sigma."""
        r_min = 1.0e-8
        r_max = max(float(r_max), 10.0 * r_min)
        radii = np.geomspace(r_min, r_max, self.n_residual)
        if residual:
            values = np.array([
                self._residual_surface_density(radius, lnM, z)
                for radius in radii
            ])
        else:
            values = np.array([
                self.surface_density(radius, lnM, z) for radius in radii
            ])
        return np.log(radii), PchipInterpolator(
            np.log(radii), values, extrapolate=False)

    def _residual_interpolator(self, r_max, lnM, z):
        """Build a signed log-radius interpolator for residual Sigma."""
        return self._surface_interpolator(r_max, lnM, z, residual=True)

    def _centered_surface_interpolator(self, r_max, lnM, z):
        """Build a log-radius interpolator for complete centered Sigma."""
        return self._surface_interpolator(r_max, lnM, z, residual=False)

    def _gamma_interpolated_surface_density(
            self, radius, r_mis, log_r, interp):
        """Gamma-convolve an interpolated centered surface density."""
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

    def _gamma_interpolated_excess_surface_density(
            self, radius, r_mis, log_r, interp):
        """Aperture-project a gamma-convolved interpolated Sigma."""
        radius = max(float(radius), 1.0e-8)
        x = radius * self._aperture_t**2
        values = np.array([
            self._gamma_interpolated_surface_density(
                value, r_mis, log_r, interp)
            for value in x
        ])
        mean = 4.0 * np.sum(
            self._aperture_w * self._aperture_t**3 * values)
        edge = self._gamma_interpolated_surface_density(
            radius, r_mis, log_r, interp)
        return float(mean - edge)

    def _gamma_residual_excess_surface_density(
            self, radius, r_mis, log_r, interp):
        """Aperture-project the gamma-convolved residual Sigma."""
        return self._gamma_interpolated_excess_surface_density(
            radius, r_mis, log_r, interp)

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
                if self.f_mis != 0.0 and self.miscentering_method == "direct":
                    log_r, residual = self._centered_surface_interpolator(
                        np.max(r_perp * q) + self.q_max * r_mis_base * q,
                        mass, redshift)
                elif (self.use_nfw_table_residual and
                      self.f_mis != 0.0):
                    log_r, residual = self._residual_interpolator(
                        np.max(r_perp * q) + self.q_max * r_mis_base * q,
                        mass, redshift)
                for ir, R in enumerate(r_perp):
                    q = 1.0 + redshift if physical else 1.0
                    centered_com = self.excess_surface_density(
                        R * q, mass, redshift)
                    centered = centered_com * q**2
                    if self.miscentering_method == "direct":
                        if self.f_mis == 0.0:
                            out[ir, ik, iz] = centered
                            continue
                        d_mis = self._gamma_interpolated_excess_surface_density(
                            R * q, r_mis_base * q, log_r, residual) * q**2
                        out[ir, ik, iz] = (
                            centered + self.f_mis * (d_mis - centered))
                        continue
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
