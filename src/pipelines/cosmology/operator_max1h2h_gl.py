"""Max-model lensing profile by direct projection of xi_max — no approximations.

For every halo (lnM, z):

    xi_max(r)     = max(xi_nfw(r, lnM), b(lnM, z) * xi_hh(r, z))
    Sigma(R)      = 2 rho_m int_0^l_max xi_max(sqrt(R^2 + l^2)) dl
    barSigma(<R)  = (2 / R^2) int_0^R R' Sigma(R') dR'
    DeltaSigma(R) = barSigma(<R) - Sigma(R)

Both integrals use fixed Gauss-Legendre nodes:

- line of sight: l = R sinh(u), u in [0, acosh(r_max / R)], dl = r du with
  r = R cosh(u), so the integral stops at the 3D radius r_max.
- aperture: R' = R t^2, t in [0, 1], so (2/R^2) R' dR' = 4 t^3 dt.

Inputs are vectorized 2D interpolators, e.g. shared.datablock_models.Bilinear2D:

    xi_nfw(r, lnM)  -- 1-halo correlation function
    bias(lnM, z)    -- Bilinear2D(source, "halomodel", "lnm", "z", "bias")
    xi_hh(r, z)     -- Bilinear2D(source, "xi_nl", "r", "z", "xi_nl")

Each must broadcast its two arguments.  Outputs have shape
(n_R, n_lnM, n_z).  rho_m is the mean matter density (M_sun h^2 / Mpc^3);
`unit` = 1e-12 returns M_sun h / pc^2, the haloModel Sigma/dSigma convention.
"""
import numpy as np

DEFAULT_N_LOS = 64
DEFAULT_N_APERTURE = 32
UNIT_MPC2_TO_PC2 = 1.0e-12


class OperatorMax1h2h_gl:
    """Sigma, barSigma and DeltaSigma of max(xi_nfw, b xi_hh) on GL nodes."""

    def __init__(self, xi_nfw, bias, xi_hh, rho_m, r_max,
                 n_los=DEFAULT_N_LOS, n_aperture=DEFAULT_N_APERTURE,
                 unit=UNIT_MPC2_TO_PC2):
        self.xi_nfw = xi_nfw
        self.bias = bias
        self.xi_hh = xi_hh
        self.rho_m = float(rho_m)
        self.r_max = float(r_max)
        self.unit = float(unit)
        self._t_los, self._w_los = np.polynomial.legendre.leggauss(n_los)
        t, w = np.polynomial.legendre.leggauss(n_aperture)
        self._t_ap = 0.5 * (t + 1.0)
        self._w_ap = 0.5 * w

    def surface_density(self, R, lnM, z):
        """Sigma(R | lnM, z) = 2 rho_m int_0^l_max xi_max dl, shape (n_R, n_lnM, n_z)."""
        R = np.atleast_1d(np.asarray(R, dtype=float))
        lnM = np.atleast_1d(np.asarray(lnM, dtype=float))
        z = np.atleast_1d(np.asarray(z, dtype=float))
        t, w = self._t_los, self._w_los
        b = self.bias(lnM[:, None], z[None, :])               # (n_lnM, n_z)

        out = np.zeros((R.size, lnM.size, z.size))
        for i, R_i in enumerate(R):
            if R_i >= self.r_max:
                continue
            u_max = np.arccosh(self.r_max / R_i)
            r = R_i * np.cosh(0.5 * u_max * (t + 1.0))         # (n_los,)
            one = self.xi_nfw(r[:, None], lnM[None, :])        # (n_los, n_lnM)
            two = self.xi_hh(r[:, None], z[None, :])           # (n_los, n_z)
            xi = np.maximum(one[:, :, None], b[None] * two[:, None, :])
            out[i] = 0.5 * u_max * np.tensordot(w * r, xi, axes=(0, 0))
        return 2.0 * self.rho_m * self.unit * out

    def mean_surface_density(self, R, lnM, z):
        """barSigma(<R) = int_0^1 4 t^3 Sigma(R t^2) dt, shape (n_R, n_lnM, n_z)."""
        R = np.atleast_1d(np.asarray(R, dtype=float))
        t, w = self._t_ap, self._w_ap
        R_in = (R[:, None] * t[None, :] ** 2).ravel()
        sigma = self.surface_density(R_in, lnM, z)
        sigma = sigma.reshape(R.size, t.size, *sigma.shape[1:])
        return np.tensordot(4.0 * w * t ** 3, sigma, axes=(0, 1))

    def excess_surface_density(self, R, lnM, z):
        """DeltaSigma(R | lnM, z) = barSigma(<R) - Sigma(R), shape (n_R, n_lnM, n_z)."""
        return (self.mean_surface_density(R, lnM, z) -
                self.surface_density(R, lnM, z))
