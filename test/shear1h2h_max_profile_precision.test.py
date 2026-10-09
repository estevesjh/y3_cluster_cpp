#!/usr/bin/env python3
"""Independent adaptive-quad precision and unit check for the 3D envelope."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))

from cosmology.halo_model import ct_2hTerm  # noqa: E402
from shared.lensing_profiles import Shear1h2hMaxProfile  # noqa: E402


class SyntheticSource:
    """Minimal case-insensitive source used by the profile adapters."""

    def __init__(self, values):
        self.values = values

    def array(self, section, key):
        return np.asarray(self.values[(section.lower(), key.lower())])

    def scalar(self, section, key):
        return float(self.values[(section.lower(), key.lower())])

    def has(self, section, key):
        return (section.lower(), key.lower()) in self.values


def make_source():
    rho_ref = 0.3 * 2.77533742639e11
    r = np.logspace(-2.0, 1.5, 192)
    z = np.array([0.0, 1.0])
    lnm = np.array([30.0, 36.0])
    xi = 0.25 / (1.0 + r)
    m3d = ct_2hTerm.m3d_at_r(r, xi, rho_m=rho_ref, n_gl=40)
    return SyntheticSource({
        ("halomodel", "lnm"): lnm,
        ("halomodel", "rho_m_ref"): rho_ref,
        ("halomodel", "r_sigma"): r,
        ("halomodel", "z"): z,
        ("halomodel", "bias"): np.full((2, 2), 2.0),
        ("halomodel", "concentration"): np.full(2, 4.0),
        ("halomodel", "dsigma_nfw"): np.ones((2, r.size)),
        ("halomodel", "sigma_bar_hh"): np.full((2, r.size), 2.0),
        ("halomodel", "dsigma_hh"): np.ones((2, r.size)),
        ("xi_nl", "r"): r,
        ("xi_nl", "z"): z,
        ("xi_nl", "xi_nl"): np.repeat(xi[None, :], 2, axis=0),
        ("xi_nl", "m3d_nl"): np.repeat(m3d[None, :], 2, axis=0),
    })


def quad_reference(profile, R, lnM=34.0, z=0.3):
    """Independent adaptive reference in u = asinh(y/R), not fixed GL."""
    R = max(float(R), float(profile._r_sigma[0]))
    r_trans = profile._transition_radius(lnM, z)
    bias = float(np.asarray(profile._bias(lnM, z)).item())
    dsigma_2h = bias * float(np.asarray(profile._dsigma_hh(R, z)).item())
    if R >= r_trans:
        delta_m = (float(profile._m3d_1h(r_trans, lnM))
                   - float(np.asarray(
                       profile._m3d_2h(r_trans, lnM, z)).item()))
        return dsigma_2h + delta_m / (np.pi * R**2) * 1.0e-12

    u_max = np.arccosh(r_trans / R)

    def delta_xi(u):
        r = R * np.cosh(u)
        return float(np.asarray(
            profile._xi_1h(r, lnM) - profile._xi_2h(r, lnM, z)).item())

    knots = profile._r_xi[
        (profile._r_xi > R) & (profile._r_xi < r_trans)]
    points = np.arccosh(knots / R)

    sigma_diff = quad(
        lambda u: delta_xi(u) * R * np.cosh(u),
        0.0, u_max, points=points, epsabs=1.0e-8,
        epsrel=1.0e-12, limit=400)[0]
    mantle = quad(
        lambda u: (delta_xi(u) * R * np.cosh(u)
                   * (R * np.sinh(u)
                      / (R * np.cosh(u) + R * np.sinh(u)))),
        0.0, u_max, points=points, epsabs=1.0e-8,
        epsrel=1.0e-12, limit=400)[0]
    delta_m = (float(profile._m3d_1h(R, lnM))
               - float(np.asarray(profile._m3d_2h(R, lnM, z)).item()))
    rho_ref = profile._rho_ref
    # rho_ref [Msun h^2 Mpc^-3] * length [Mpc h^-1] gives
    # Msun h Mpc^-2; 1 Mpc^-2 = 1e-12 pc^-2.
    return dsigma_2h + (
        delta_m / (np.pi * R**2)
        + 4.0 * rho_ref * mantle
        - 2.0 * rho_ref * sigma_diff
    ) * 1.0e-12


class TestShear1h2hMaxProfilePrecision(unittest.TestCase):
    def test_twenty_point_gl_matches_adaptive_quad_and_units(self):
        source = make_source()
        profile = Shear1h2hMaxProfile(
            source, lob_centers=[30.0], f_mis=0.0, tau_mis=0.17,
            omega_m=0.3, n_gl=20, n_root=256)
        radii = np.logspace(-2.0, 1.0, 12)
        reference = np.array([quad_reference(profile, R) for R in radii])
        got = np.array([
            profile.excess_surface_density(R, 34.0, 0.3) for R in radii])
        np.testing.assert_allclose(got, reference, rtol=1.0e-6,
                                   atol=2.0e-6)

    def test_surface_density_and_shear_identities(self):
        profile = Shear1h2hMaxProfile(
            make_source(), lob_centers=[30.0], f_mis=0.0, tau_mis=0.17,
            omega_m=0.3, n_gl=20, n_root=256)
        radii = np.logspace(-2.0, 1.0, 8)
        sigma = np.array([
            profile.surface_density(R, 34.0, 0.3) for R in radii])
        mean_sigma = np.array([
            profile.mean_surface_density(R, 34.0, 0.3) for R in radii])
        delta_sigma = np.array([
            profile.excess_surface_density(R, 34.0, 0.3) for R in radii])
        np.testing.assert_allclose(mean_sigma - sigma, delta_sigma,
                                   rtol=2.0e-14, atol=2.0e-14)
        sigma_crit_inv = 2.5e-4
        shear = np.array([
            profile.tangential_shear(R, 34.0, 0.3, sigma_crit_inv)
            for R in radii])
        np.testing.assert_allclose(
            shear, delta_sigma * sigma_crit_inv,
            rtol=2.0e-14, atol=2.0e-14)

    def test_nfw_table_plus_signed_residual_correction_is_finite(self):
        profile = Shear1h2hMaxProfile(
            make_source(), lob_centers=[30.0], f_mis=0.2, tau_mis=0.17,
            omega_m=0.3, n_gl=8, n_root=64, n_offset=2, n_phi=8,
            n_aperture=3, n_residual=64)
        values = profile.excess_surface_density_grid(
            0, np.array([0.2, 0.6]), np.array([34.0]), np.array([0.3]))
        self.assertEqual(values.shape, (2, 1, 1))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_direct_full_surface_density_convolution_is_finite(self):
        profile = Shear1h2hMaxProfile(
            make_source(), lob_centers=[30.0], f_mis=0.2, tau_mis=0.17,
            omega_m=0.3, n_gl=8, n_root=64, n_offset=2, n_phi=8,
            n_aperture=3, miscentering_method="direct")
        values = profile.excess_surface_density_grid(
            0, np.array([0.2, 0.6]), np.array([34.0]), np.array([0.3]))
        self.assertEqual(values.shape, (2, 1, 1))
        self.assertTrue(np.all(np.isfinite(values)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
