"""Traditional 1h+2h excess surface density via the max model — Python backend.

Python twin of ``shear_1h2h/cpp/0d/shear1h2h_max_t.hh`` (class ``Shear1h2hMax``,
same options, same DataBlock readers, same output section); the two are
interchangeable — never run both in one pipeline.

Formulated via 3D halo exclusion:

    xi_hm(r) = max(xi_1h(r), b(M, z) * xi_mm(r, z))

profile_method selects how DeltaSigma_max(R | lnM, z) is computed:
  piecewise_approx (default): around the transition radius r_trans,
      R <  r_trans : DeltaSigma_1h(R) + Delta_gap^1 (R / r_trans)^2
      R >= r_trans : b DeltaSigma_hh(R) + (C_0 / R^2) * 1e-12
    with r_trans, c0_trans, delta_gap_1 read from halomodel.
  gauss_legendre: exact GL projection of xi_max
    (cosmology.operator_max1h2h_gl.OperatorMax1h2h_gl), with
    xi_nfw = rho_NFW / rho_m - 1 from halomodel/concentration and
    rho_m = halomodel/rho_m_ref, and xi_nl read with a bicubic spline.
    The profile is bin-independent, so set_sample tabulates Sigma at every
    r_perp radius and its aperture nodes once.

Stacked with the selection weight

    O_ij(R) = sum_{k,q} W2d(b; lnM_k, z_q) DeltaSigma(R | lnM_k, z_q),
    W2d     = w_k^lnM w_q^z dV/dz dOmega Omega Sigma_crit^-1 dn/dlnM S_b.

The output is the stacked numerator O_ij(R) (not normalised).

Optional bsel (bsel_model = Costanzi26 or Sunayama23), for a given (lnM, z):
  piecewise_approx: BselDeltaSigmaOperator, B DeltaSigma - M_sel/(pi R^2),
    which additionally requires halomodel sigma_bar_nfw, sigma_bar_hh and
    delta_gap_0 for barSigma(<r).
  gauss_legendre: the equivalent local form
    (2/R^2) int_0^R R' B Sigma dR' - B(R) Sigma(R), i.e. B weights the
    aperture nodes — no extra tables.

Options: bin_index x r_perp cartesian grid (bin slow / R fast),
zt_low/zt_high/lnm_low/lnm_high (required), n_lnm (96), n_z (64),
lob_centers (bsel richness anchor), bsel (default F), bsel_model,
profile_method (piecewise_approx), n_los (64), n_aperture (32).
Output: shear1h2h_max/vals  (n_bins * n_r,)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

for _p in Path(__file__).resolve().parents:
    if (_p / "shared" / "datablock_models.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from cosmology.operator_max1h2h_gl import OperatorMax1h2h_gl  # noqa: E402
from shared import datablock_models as dm                     # noqa: E402
from systematics.selection_richness.python import sel_kernels  # noqa: E402
from systematics.selection_boost.Bsel import BselDeltaSigmaOperator  # noqa: E402

OUTPUT_SECTION = "shear1h2h_max"
PROFILE_METHODS = ("piecewise_approx", "gauss_legendre")


class Shear1h2hMax:
    """Stacked max-model DeltaSigma; mirrors the C++ Shear1h2hMax."""

    def __init__(self, options, section):
        def opt(get, key, default):
            return get(section, key) if options.has_value(section, key) else default

        sf = sel_kernels.load()
        self.bin_index = sf._read_array(options, section, "bin_index").astype(int)
        self.r_perp = sf._read_array(options, section, "r_perp")
        self.n_lnm = int(opt(options.get_int, "n_lnm", 96))
        self.n_z = int(opt(options.get_int, "n_z", 64))
        self.zt_lo = options.get_double(section, "zt_low")
        self.zt_hi = options.get_double(section, "zt_high")
        self.lnm_lo = options.get_double(section, "lnm_low")
        self.lnm_hi = options.get_double(section, "lnm_high")
        self.bsel_enabled = bool(opt(options.get_bool, "bsel", False))
        self.bsel_model = (options.get_string(section, "bsel_model")
                           if self.bsel_enabled else "")
        self.profile_method = opt(options.get_string, "profile_method",
                                  "piecewise_approx")
        self.n_los = int(opt(options.get_int, "n_los", 64))
        self.n_aperture = int(opt(options.get_int, "n_aperture", 32))

        self.lnm_x, self.lnm_w = dm.gl_nodes(self.lnm_lo, self.lnm_hi, self.n_lnm)
        self.z_x, self.z_w = dm.gl_nodes(self.zt_lo, self.zt_hi, self.n_z)
        self.lob_centers = (sf._read_array(options, section, "lob_centers")
                            if options.has_value(section, "lob_centers")
                            else np.asarray(dm.DEFAULT_LOB_CENTERS, dtype=float))
        if self.lob_centers.size == 0:
            raise RuntimeError("Shear1h2hMax: lob_centers is empty")
        if self.profile_method not in PROFILE_METHODS:
            raise ValueError("Shear1h2hMax: unknown profile_method")

    def set_sample(self, s):
        self._bsel_op = (BselDeltaSigmaOperator.from_source(s, self.bsel_model)
                         if self.bsel_enabled else None)
        hmf = dm.HMF(s)
        dv = dm.DVDoDz(s)
        sci = dm.SigmaCritInv(s)
        sel = dm.SelStack(s)
        bias = dm.Bilinear2D(s, "halomodel", "lnm", "z", "bias")

        # z-only factors (Sigma_crit_inv folded in, as the shear weight
        # requires) and the z-RESOLVED weight W2d(bin; lnM, z).
        zfac = self.z_w * dv(self.z_x) * dm.omega_z_des(self.z_x) * sci(self.z_x)

        # W2d(b; lnM_k, z_q) = w_k^lnM * zfac(z_q) * dn/dlnM(M_k, z_q)
        #                      * S_b(M_k, z_q),
        # with S_b the probability a halo of (M, z) lands in observed bin b.
        # R-independent, so built once per sample; evaluate() then reduces the
        # (lnM, z) integral to acc = sum_{k,q} W2d * DeltaSigma(R | lnM_k, z_q).
        # Layout (b, k, q).  Not normalised here.
        lnm, z = self.lnm_x[:, None], self.z_x[None, :]
        base = self.lnm_w[:, None] * zfac[None, :] * hmf(lnm, z)
        self.n_bins = sel.n_bins
        self.w2d = np.stack([base * sel(b, lnm, z) for b in range(self.n_bins)])

        if self.profile_method == "gauss_legendre":
            self._set_sample_gauss_legendre(s, bias)
        else:
            self._set_sample_piecewise(s, bias)

    def evaluate(self, b, R):
        if b < 0 or b >= self.n_bins:
            raise IndexError("Shear1h2hMax: bin_index out of range")
        lob = self.lob_centers[b % self.lob_centers.size]
        if self.profile_method == "gauss_legendre":
            return self._evaluate_gauss_legendre(self.w2d[b], b, R, lob)
        return self._evaluate_piecewise(self.w2d[b], b, R, lob)

    # ---- piecewise_approx -------------------------------------------------

    def _set_sample_piecewise(self, s, bias):
        self._sigma_bar_nfw = self._sigma_bar_hh = None
        if self.bsel_enabled:
            self._sigma_bar_nfw = dm.Bilinear2D(
                s, "halomodel", "r_sigma", "lnm", "sigma_bar_nfw")
            self._sigma_bar_hh = dm.Bilinear2D(
                s, "halomodel", "r_sigma", "z", "sigma_bar_hh", nan_fill=0.0)
        self._dsigma_nfw = dm.Bilinear2D(s, "halomodel", "r_sigma", "lnm", "dsigma_nfw")
        self._dsigma_hh = dm.Bilinear2D(s, "halomodel", "r_sigma", "z", "dsigma_hh",
                                        nan_fill=0.0)
        # Piecewise max-model terms: required, no max(1h, 2h) fallback.
        # (lnM, z)-only terms on the node grid — bin- and R-independent.
        lnm, z = self.lnm_x[:, None], self.z_x[None, :]
        self._bias_kq = bias(lnm, z)
        self._r_trans_kq = dm.Bilinear2D(s, "halomodel", "lnm", "z", "r_trans")(lnm, z)
        self._c0_trans_kq = dm.Bilinear2D(s, "halomodel", "lnm", "z", "c0_trans")(lnm, z)
        self._delta_gap_1_kq = dm.Bilinear2D(s, "halomodel", "lnm", "z", "delta_gap_1")(lnm, z)
        self._delta_gap_0_kq = (
            dm.Bilinear2D(s, "halomodel", "lnm", "z", "delta_gap_0")(lnm, z)
            if self.bsel_enabled else np.zeros_like(self._bias_kq))

    def _evaluate_piecewise(self, w2, b, R, lob):
        # Per-(bin, R) profile rows: the 1-halo term is z-free (one value per
        # mass node), the 2-halo table row is mass-free (one per z node).
        one = self._dsigma_nfw(R, self.lnm_x)[:, None]       # (k, 1)
        two = self._dsigma_hh(R, self.z_x)[None, :]          # (1, q)
        d_max = self._excess_surface_density_node(R, one, two)
        if self._bsel_op is not None:
            # For a given (lnM, z): B_sel(R) d_max - M_sel(R) / (pi R^2), with
            # barSigma(<r) from the same piecewise profile.
            d_max = self._bsel_op.apply(
                d_max, self._mean_surface_density_node, R,
                bin_index=b, lob=lob, z=self.z_x[None, :])
        return float(np.sum(w2 * d_max))

    def _excess_surface_density_node(self, R, one, two):
        """Piecewise DeltaSigma_max(R) on the (lnM, z) node grid:
          R <  r_trans : DeltaSigma_1h(R) + Delta_gap^1 (R / r_trans)^2
          R >= r_trans : b DeltaSigma_hh(R) + C_0 / R^2
        """
        rt = self._r_trans_kq
        outer = self._bias_kq * two + (self._c0_trans_kq / (R * R)) * 1.0e-12
        inner = one + self._delta_gap_1_kq * (R / rt) ** 2
        return np.where(R >= rt, outer, inner)

    def _mean_surface_density_node(self, r):
        """barSigma(<r) = Sigma(r) + DeltaSigma(r) of the same piecewise
        profile on the (lnM, z) node grid; the mean_sigma_fn input to
        BselDeltaSigmaOperator.apply.  Only built when bsel is enabled.
          r <  r_trans : barSigma_1h(r) + Delta_gap^0 + 2 Delta_gap^1
                         - Delta_gap^1 (r / r_trans)^2
          r >= r_trans : b barSigma_hh(r) + C_0 / r^2
        """
        rt = self._r_trans_kq
        gap1 = self._delta_gap_1_kq
        outer = (self._bias_kq * self._sigma_bar_hh(r, self.z_x)[None, :]
                 + (self._c0_trans_kq / (r * r)) * 1.0e-12)
        inner = (self._sigma_bar_nfw(r, self.lnm_x)[:, None] + self._delta_gap_0_kq
                 + 2.0 * gap1 - gap1 * (r / rt) ** 2)
        return np.where(r >= rt, outer, inner)

    # ---- gauss_legendre ---------------------------------------------------

    def _set_sample_gauss_legendre(self, s, bias):
        # xi_nfw(r, lnM) = rho_NFW / rho_m - 1 with M = M200m, the same
        # convention halo_model uses for xi_1h (rho_s / rho_m = delta_c).
        rho_m = s.scalar("halomodel", "rho_m_ref")
        conc_lnm = s.array("halomodel", "lnm")
        conc = s.array("halomodel", "concentration")

        def xi_nfw(r, lnM):
            c = np.interp(lnM, conc_lnm, conc)
            r200 = np.cbrt(3.0 * np.exp(lnM) / (800.0 * np.pi * rho_m))
            x = r * c / r200
            delta_c = (200.0 / 3.0) * c ** 3 / (np.log1p(c) - c / (1.0 + c))
            return delta_c / (x * (1.0 + x) ** 2) - 1.0

        # Bicubic: bilinear kinks at every r knot stall the GL convergence.
        xi_nl = dm.Bilinear2D(s, "xi_nl", "r", "z", "xi_nl", cubic=True)
        r_max = s.array("xi_nl", "r")[-1]
        self._gl_op = OperatorMax1h2h_gl(
            xi_nfw, bias, xi_nl, rho_m, r_max,
            n_los=self.n_los, n_aperture=self.n_aperture)
        self._gl_aperture_weights = self._gl_op.aperture_weights()

        # Bin-independent: tabulate every wall radius once per sample.
        self._gl_sigma = {float(R): self._gl_sigma_slots(R) for R in self.r_perp}

    def _gl_sigma_slots(self, R):
        """Sigma on the (lnM, z) grid at R (slot 0) and at the aperture radii
        R t_j^2 (slot j + 1): shape (n_aperture + 1, n_lnM, n_z)."""
        t = self._gl_op.aperture_nodes()
        radii = np.concatenate(([R], R * t * t))
        return self._gl_op.surface_density(radii, self.lnm_x, self.z_x)

    def _evaluate_gauss_legendre(self, w2, b, R, lob):
        """DeltaSigma = sum_j a_j B(R t_j^2) Sigma(R t_j^2) - B(R) Sigma(R),
        with a_j = 4 w_j t_j^3 and B = 1 without bsel.  With bsel this is
        exactly B DeltaSigma - M_sel / (pi R^2) (integration by parts)."""
        sig = self._gl_sigma.get(float(R))
        if sig is None:
            sig = self._gl_sigma_slots(R)
        t = self._gl_op.aperture_nodes()
        radii = np.concatenate(([R], R * t * t))
        coef = np.concatenate(([-1.0], self._gl_aperture_weights))[:, None]
        if self._bsel_op is not None:
            B = self._bsel_op.model(radii[:, None], bin_index=b, lob=lob,
                                    z=self.z_x[None, :])
            coef = coef * np.broadcast_to(B, (radii.size, self.z_x.size))
        coef = np.broadcast_to(coef, (radii.size, self.z_x.size))
        return float(np.einsum("sq,skq,kq->", coef, sig, w2))

    @staticmethod
    def module_label():
        return "Shear1h2hMax"


def setup(options):
    from cosmosis.datablock import option_section
    return Shear1h2hMax(options, option_section)


def execute(block, mod):
    t0 = time.perf_counter()
    mod.set_sample(dm.DataBlockSource(block))
    # Cartesian bin_index x r_perp wall, bin slow / R fast.
    vals = np.array([mod.evaluate(int(b), float(R))
                     for b in mod.bin_index for R in mod.r_perp])
    block[OUTPUT_SECTION, "vals"] = vals
    dt_ms = 1000.0 * (time.perf_counter() - t0)
    print(f"[shear1h2h_max] {mod.bin_index.size} bins x {mod.r_perp.size} "
          f"radii ({mod.profile_method}) — {dt_ms:.0f} ms", flush=True)
    return 0


def cleanup(config):
    return 0
