"""Projection shear via the fast_mass strategy — Python (exact z).

The two-halo term sourced by correlated line-of-sight structure with the
selection-affected bias b_sel(theta), computed with the redshift
integral done exactly, outside the radial operator — the fast_mass
pattern extended to the projection observable at the plan owner's
direction. Per wall point (lambda_ob bin, z_ob, R):

    DSigma_prj(R) = int dtheta 2 pi sin(theta) [
        Sum_M wrnd(M)                DSmis(R, theta D_A | M)      (rnd)
      + b_sel(theta) Sum_M wcl(theta, M) DSmis(R, theta D_A | M)  (cl) ]
    gamma_t^prj = DSigma_prj * <Sigma_crit^-1>(z_ob)

with the exact per-slice redshift weights

    wrnd(M)       = int dz  common(z) n(M, z)
    wcl(theta, M) = int dz  common(z) xi_NL(|dchi|(z, theta), z_ob)
                            n(M, z) b(M, z) 1[theta > theta_excl(z)]
    common(z)     = dV/dOmega/dz(z) w_phot(z; z_ob) w_z^GL

This is a convention-exact Python port of the *exact* C++ core
(sp_detail::ShearPrjCore in src/pipelines/systematics/shear_prj/cpp/
sigma_prj_t.hh): identical theta
grid (per-slice breakpoints + log-GL segments), identical z grid
(exclusion ring + foreground/background log-|dchi| wings via the same
40-iteration chi inversion), the parabolic photo-z weight with the
compiled sigma_z table, the analytic Costanzi-2026 b_sel sigmoid on the
(b_small, b_large) plateaus, the single-offset NFW table with the
rho_mult = Omega_m amplitude, and no Omega(z) (it cancels in the
surface density — the exact core hard-excludes it). The production
`shear_prj_frozen_physics` module additionally freezes the cl-channel
mass shape at z_ob; this port does not (no frozen-physics
approximation), so it validates against the exact ShearPrjEvaluator at
machine precision and against the frozen production module at the
documented <0.2% level.

DataBlock contract
------------------
Reads (ini options, ShearPrjCore conventions): the 180-point zipped
wall lambda_bin/zo_low/zo_high/radii; zt_low/zt_high,
lnm_low/lnm_high; n_lnm (16), n_per_seg (10), n_zring (20),
n_zouter (20), R_max_cMpch (35); lob_centers (default 25 37.5 52.5 130).
Reads (datablock): mass_function/* + cluster_abundance/{hmf_s,hmf_q},
halomodel/{lnm,z,bias}, xi_nl/{r,z,xi_nl}, distances/{z,d_a,d_c},
b_sel_marginalised/{lob,zob,b_small,b_large},
average_sigma_crit_inv/{zlense,sci_average},
cosmological_parameters/{h0,omega_m,omega_nu,omega_lambda,omega_k},
data/nfw_off_center/*single* (fixed tables), and the compiled
z_kernel_data.hh sigma_z table.
Writes (hardcoded sections):
    dsigma_prj_fast_mass/{vals,rnd,cl}   Msun/(h pc^2), length 180
    shear_prj_fast_mass/{vals,rnd,cl}    dimensionless, length 180

Status: reference implementation (exact-z fast path). Production
remains shear_prj_frozen_physics (frozen cl channel).
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

from shared import datablock_models as dm
from shared import lensing_profiles as lp
from systematics.selection_richness.python import sel_kernels
from shared import z_kernel

DSIGMA_SECTION = "dsigma_prj_fast_mass"
SHEAR_SECTION = "shear_prj_fast_mass"


def build_theta_grid(lobc, zob, r_vec, chi_o, d_a_o, r_excl, n_per_seg,
                     r_max_cmpch):
    """Port of sp_detail::build_theta_grid (breakpoints + log-GL)."""
    theta_lam = lp.r_lambda(lobc) * (1.0 + zob) / chi_o
    theta_excl_o = r_excl / chi_o
    theta_r = np.asarray(r_vec, dtype=float) / d_a_o
    theta_r_min = theta_r.min() if theta_r.size else theta_lam
    theta_r_max = theta_r.max() if theta_r.size else theta_lam
    theta_max = max(r_max_cmpch / d_a_o, 3.0 * theta_r_max)
    lower = max(1.0e-8, 0.1 * min(theta_excl_o, theta_r_min, theta_lam))

    bp = np.concatenate([[lower, theta_excl_o], theta_r,
                         [theta_lam, 2.0 * theta_lam, theta_max]])
    bp = np.sort(bp[np.isfinite(bp) & (bp > 0.0) & (bp <= theta_max)])
    if bp.size == 0 or bp[0] > lower:
        bp = np.concatenate([[lower], bp])
    if bp[-1] < theta_max:
        bp = np.concatenate([bp, [theta_max]])
    dedup = [bp[0]]
    for t in bp[1:]:
        if t > dedup[-1] * (1.0 + 1.0e-6):
            dedup.append(t)

    theta, weight = [], []
    for a, b in zip(dedup[:-1], dedup[1:]):
        u_x, u_w = dm.gl_nodes(np.log(a), np.log(b), n_per_seg)
        th = np.exp(u_x)
        theta.append(th)
        weight.append(u_w * th)
    return np.concatenate(theta), np.concatenate(weight)


def theta_excl_at_z(chi_z, chi_o, r_excl):
    """Port of sp_detail::theta_excl_at_z (vectorised over chi_z)."""
    cos_ex = ((chi_z**2 + chi_o**2 - r_excl**2)
              / (2.0 * chi_z * chi_o + 1.0e-30))
    out = np.arccos(np.clip(cos_ex, -1.0, None).clip(max=1.0))
    return np.where(cos_ex >= 1.0 - 1.0e-12, 0.0, out)


class ShearPrjFastMass:
    """Exact-z projection shear on the production wall grid."""

    def __init__(self, wall, *, lob_centers, n_lnm=16, n_per_seg=10,
                 n_zring=20, n_zouter=20, zt_low=0.10, zt_high=0.75,
                 lnm_low=29.9336, lnm_high=35.6814, r_max_cmpch=35.0):
        self.cfg = dict(n_lnm=n_lnm, n_per_seg=n_per_seg, n_zring=n_zring,
                        n_zouter=n_zouter, zt_low=zt_low, zt_high=zt_high,
                        r_max_cmpch=r_max_cmpch)
        self.lob_centers = np.asarray(lob_centers, dtype=float)
        self.lnm_x, self.lnm_w = dm.gl_nodes(lnm_low, lnm_high, n_lnm)

        # Dedupe the wall into (lob_bin, zob) slices, keeping each
        # slice's radii in wall order (ShearPrjCore ctor semantics).
        lb = np.asarray(wall["lambda_bin"], dtype=int)
        zob = 0.5 * (np.asarray(wall["zo_low"], dtype=float)
                     + np.asarray(wall["zo_high"], dtype=float))
        radii = np.asarray(wall["radii"], dtype=float)
        self.wall_n = lb.size
        self.slices = []          # list of dict(lb, zob, Rs, wall_idx)
        keymap = {}
        for i in range(lb.size):
            key = (int(lb[i]), float(zob[i]))
            if key not in keymap:
                keymap[key] = len(self.slices)
                self.slices.append(dict(lb=int(lb[i]), zob=float(zob[i]),
                                        Rs=[], wall_idx=[]))
            s = self.slices[keymap[key]]
            s["Rs"].append(float(radii[i]))
            s["wall_idx"].append(i)

    # -- per-sample ------------------------------------------------------

    def set_sample(self, source):
        cfg = self.cfg
        self._dist_z = source.array("distances", "z")
        self._d_c = source.array("distances", "d_c")
        self._h0 = source.scalar("cosmological_parameters", "h0")
        omega_m = source.scalar("cosmological_parameters", "omega_m")

        hmf = dm.HMF(source)
        dv = dm.DVDoDz(source)
        bias = dm.Bilinear2D(source, "halomodel", "lnm", "z", "bias")
        xi_nl = dm.Bilinear2D(source, "xi_nl", "r", "z", "xi_nl")
        sci = dm.SigmaCritInv(source)
        dsmis = lp.NfwDsigmaMisProduction(kernel="single")

        bs_lob = np.asarray(source.array("b_sel_marginalised", "lob"),
                            dtype=float)
        bs_zob = np.asarray(source.array("b_sel_marginalised", "zob"),
                            dtype=float)

        # The CosmoSIS datablock stores one row per (zob, lob) wall point,
        # while a few offline sources store separate zob/lob axes.  Convert
        # either representation to the table consumed below:
        # b_sel[zob_index, lob_index].
        zob_nodes = np.unique(bs_zob)
        lob_nodes = np.unique(bs_lob)

        def bsel_table(name):
            values = np.asarray(source.array("b_sel_marginalised", name),
                                dtype=float).ravel()
            if values.size == zob_nodes.size * lob_nodes.size \
                    and values.size != bs_zob.size:
                return values.reshape(zob_nodes.size, lob_nodes.size)
            if values.size != bs_zob.size:
                raise ValueError(
                    f"b_sel_marginalised/{name} has {values.size} values "
                    f"for {bs_zob.size} wall rows")
            table = np.full((zob_nodes.size, lob_nodes.size), np.nan)
            for row, (zob_row, lob_row) in enumerate(zip(bs_zob, bs_lob)):
                iz = np.flatnonzero(np.isclose(zob_nodes, zob_row))[0]
                il = np.flatnonzero(np.isclose(lob_nodes, lob_row))[0]
                table[iz, il] = values[row]
            if not np.all(np.isfinite(table)):
                raise ValueError(
                    f"b_sel_marginalised/{name} does not cover every "
                    "(zob, lob) combination")
            return table

        b_small = bsel_table("b_small")
        b_large = bsel_table("b_large")
        bs_zob = np.unique(bs_zob)
        bs_lob = np.unique(bs_lob)

        chi = lambda z: np.interp(np.clip(z, self._dist_z[0],
                                          self._dist_z[-1]),
                                  self._dist_z, self._d_c) * self._h0

        self._results = {}
        for s in self.slices:
            lobc = float(self.lob_centers[s["lb"]])
            zob = s["zob"]
            chi_o = float(chi(zob))
            d_a_o = chi_o / (1.0 + zob)
            r_excl = float(lp.r_lambda(lobc)) * (1.0 + zob)
            sci_o = float(sci(zob))

            theta, w_theta = build_theta_grid(
                lobc, zob, s["Rs"], chi_o, d_a_o, r_excl,
                cfg["n_per_seg"], cfg["r_max_cmpch"])
            geom = w_theta * 2.0 * np.pi * np.sin(theta)

            # b_sel(theta): plateau interpolation in zob + sigmoid.
            # Bracket-search port of sigma_prj_t.hh:829-851
            # (interp_b_asymptotes_) — finds the bs_zob table nodes
            # bracketing this slice's zob; unrelated to the theta grid.
            j = 0
            while j + 1 < bs_zob.size and bs_zob[j + 1] < zob:
                j += 1
            j0 = j if j + 1 < bs_zob.size else bs_zob.size - 2
            j1 = j0 + 1
            f = np.clip((zob - bs_zob[j0]) / (bs_zob[j1] - bs_zob[j0])
                        if bs_zob[j1] > bs_zob[j0] else 0.0, 0.0, 1.0)
            bs = (1 - f) * b_small[j0, s["lb"]] + f * b_small[j1, s["lb"]]
            bl = (1 - f) * b_large[j0, s["lb"]] + f * b_large[j1, s["lb"]]
            theta_lam = float(lp.r_lambda(lobc)) * (1.0 + zob) / chi_o
            k_sig, theta0 = 2.5 / theta_lam, 0.5 * theta_lam
            bsel = bs + (bl - bs) / (1.0 + np.exp(-k_sig * (theta - theta0)))

            zs, wzs = self._build_z_grid(zob, chi, r_excl)

            # Exact z contraction (the fast_mass step).
            chi_z = chi(zs)
            sig_z = z_kernel.sigma_z(zs)
            u_phot = (zs - zob) / sig_z
            w_phot = np.where(np.abs(u_phot) < 1.0, 1.0 - u_phot**2, 0.0)
            common = dv(zs) * wzs * w_phot                     # (nz,)
            hmf_zm = hmf(self.lnm_x[None, :], zs[:, None])     # (nz, nm)
            bias_zm = bias(self.lnm_x[None, :], zs[:, None])
            wrnd = (common[:, None] * hmf_zm * self.lnm_w[None, :]).sum(0)

            th_excl = theta_excl_at_z(chi_z, chi_o, r_excl)    # (nz,)
            mask = theta[None, :] > th_excl[:, None]           # (nz, nth)
            dchi = np.sqrt(np.maximum(
                chi_z[:, None]**2 + chi_o**2
                - 2.0 * chi_z[:, None] * chi_o * np.cos(theta)[None, :],
                0.0))
            xi = xi_nl(dchi, zob) * mask
            wcl = np.einsum("zt,zm->tm", common[:, None] * xi,
                            hmf_zm * bias_zm * self.lnm_w[None, :])

            # Per-R NFW cache and the theta x M dot products.
            rnd_R, cl_R = [], []
            for r_perp in s["Rs"]:
                ds = dsmis(r_perp, theta[:, None] * d_a_o,
                           self.lnm_x[None, :], rho_mult=omega_m)
                rnd_R.append(geom @ (ds @ wrnd))
                cl_R.append((geom * bsel) @ np.sum(wcl * ds, axis=1))
            self._results[(s["lb"], zob)] = dict(
                rnd=np.asarray(rnd_R), cl=np.asarray(cl_R), sci=sci_o,
                wall_idx=s["wall_idx"])

    def _build_z_grid(self, zob, chi, r_excl):
        """Port of ShearPrjCore::build_z_grid_ (ring + log-|dchi| wings)."""
        cfg = self.cfg
        z_lo, z_hi = cfg["zt_low"], cfg["zt_high"]
        chi_fg_lo, chi_bg_hi = float(chi(z_lo)), float(chi(z_hi))
        chi_o = float(chi(zob))
        dz = 1.0e-3
        dchi_dz = (float(chi(zob + dz)) - float(chi(zob - dz))) / (2 * dz)
        dz_excl = r_excl / dchi_dz
        ring_lo, ring_hi = max(zob - dz_excl, z_lo), min(zob + dz_excl, z_hi)

        def invert_chi(target):
            lo, hi = 0.001, 2.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if float(chi(mid)) < target:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        parts = []
        if ring_hi > ring_lo:
            parts.append(dm.gl_nodes(ring_lo, ring_hi, cfg["n_zring"]))
        for sign, dis_max in ((-1, chi_o - chi_fg_lo),
                              (+1, chi_bg_hi - chi_o)):
            if r_excl < dis_max:
                u_x, u_w = dm.gl_nodes(np.log(r_excl), np.log(dis_max),
                                       cfg["n_zouter"])
                dis = np.exp(u_x)
                z_i = np.array([invert_chi(chi_o + sign * d) for d in dis])
                ddz = np.array([(float(chi(z + dz)) - float(chi(z - dz)))
                                / (2 * dz) for z in z_i])
                w_i = u_w * dis / ddz
                # C++ appends fg reversed, then ring, then bg.
                order = slice(None, None, -1) if sign < 0 else slice(None)
                parts.insert(0 if sign < 0 else len(parts),
                             (z_i[order], w_i[order]))
        zs = np.concatenate([p[0] for p in parts])
        wzs = np.concatenate([p[1] for p in parts])
        return zs, wzs

    def wall_outputs(self):
        """(dsigma_rnd, dsigma_cl, shear factor) flattened to the wall."""
        rnd = np.empty(self.wall_n)
        cl = np.empty(self.wall_n)
        sci = np.empty(self.wall_n)
        for res in self._results.values():
            idx = res["wall_idx"]
            rnd[idx] = res["rnd"]
            cl[idx] = res["cl"]
            sci[idx] = res["sci"]
        return rnd, cl, sci


# ---------------------------------------------------------------------------
# CosmoSIS module entry points
# ---------------------------------------------------------------------------

def setup(options):
    from cosmosis.datablock import option_section
    sf = sel_kernels.load()
    wall = {key: sf._read_array(options, option_section, key)
            for key in ("lambda_bin", "zo_low", "zo_high", "radii")}
    try:
        lob_centers = sf._read_array(options, option_section, "lob_centers")
    except Exception:
        lob_centers = np.asarray(dm.DEFAULT_LOB_CENTERS)
    knobs = {}
    for key, default in (("n_lnm", 16), ("n_per_seg", 10),
                         ("n_zring", 20), ("n_zouter", 20)):
        try:
            knobs[key] = int(options.get_int(option_section, key))
        except Exception:
            knobs[key] = default
    for key, default in (("zt_low", 0.10), ("zt_high", 0.75),
                         ("lnm_low", 29.9336), ("lnm_high", 35.6814),
                         ("r_max_cmpch", 35.0)):
        try:
            knobs[key] = float(options.get_double(
                option_section, "R_max_cMpch" if key == "r_max_cmpch"
                else key))
        except Exception:
            knobs[key] = default
    return ShearPrjFastMass(wall, lob_centers=lob_centers, **knobs)


def execute(block, core):
    t0 = time.perf_counter()
    core.set_sample(dm.DataBlockSource(block))
    rnd, cl, sci = core.wall_outputs()
    block[DSIGMA_SECTION, "rnd"] = rnd
    block[DSIGMA_SECTION, "cl"] = cl
    block[DSIGMA_SECTION, "vals"] = rnd + cl
    block[SHEAR_SECTION, "rnd"] = rnd * sci
    block[SHEAR_SECTION, "cl"] = cl * sci
    block[SHEAR_SECTION, "vals"] = (rnd + cl) * sci
    dt_ms = 1000.0 * (time.perf_counter() - t0)
    print(f"[shear_prj_fast_mass] {core.wall_n} wall points, "
          f"{len(core.slices)} slices — {dt_ms:.0f} ms", flush=True)
    return 0


def cleanup(config):
    return 0
