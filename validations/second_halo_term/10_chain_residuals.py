#!/usr/bin/env python
"""Per-code residuals against the closed-form NFW chain: cluster_toolkit,
CLensPy, and CLMM, each compared to the analytic rho/xi, Sigma(R) and
DeltaSigma(R) (Wright & Brainerd 2000) of the same NFW halo
(M200m = 1e14 Msun/h, c = 5, comoving Omega_{m,0} rho_{c,0} background,
z = 0 so comoving = physical).

What each code contributes per stage:
  rho/xi : cluster_toolkit xi_mm_at_r and CLensPy's FFTLog wrapper are fed
           the closed-form 3-D Fourier transform and must return rho(r)
           (P->xi convention = same integral); CLMM has no generic
           transform, so its native NFW 3-D density is compared instead
           (validates its conventions and our closed forms).
  Sigma  : cluster_toolkit Sigma_at_R (Abel of rho/rho_m), CLensPy
           compute_sigma_grid (Abel quadrature), CLMM
           compute_surface_density (native NFW).
  DSigma : cluster_toolkit DeltaSigma_at_R (native path), CLensPy
           extended-grid cumtrapz interior mean, CLMM
           compute_excess_surface_density (native NFW).

Stages (two interpreters, exchange via .npz):
  --stage ct       conda env   -> outputs/chain_res_ct.npz
  --stage clenspy  CLensPy venv (-B) -> outputs/chain_res_clenspy.npz
  --stage figure   conda env (seaborn) -> report/figs + report/tables
                   + report/values_chain_residuals.tex
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "outputs")
FIGS = os.path.join(HERE, "report", "figs")
TABLES = os.path.join(HERE, "report", "tables")
sys.path.insert(0, os.path.join(HERE, "common"))

RHO_C = 2.77533742639e11      # Msun/Mpc^3/h^2
OMEGA_M = 0.311049
OMEGA_B = 0.048975
H = 0.6766
RHO_M = OMEGA_M * RHO_C       # Msun h^2/Mpc^3, comoving, z-independent

M200M_H = 1e14                # Msun/h
CONC = 5.0

R3D = np.logspace(np.log10(0.05), np.log10(30.0), 72)    # cMpc/h
R2D = np.logspace(np.log10(0.1), np.log10(20.0), 128)    # cMpc/h


def nfw_profile():
    from analytic_profiles import NFWProfile
    r200 = (3.0 * M200M_H / (4.0 * np.pi * 200.0 * RHO_M)) ** (1.0 / 3.0)
    r_s = r200 / CONC
    delta_c = (200.0 / 3.0) * CONC ** 3 / (np.log(1 + CONC) - CONC / (1 + CONC))
    return NFWProfile(rho_s=delta_c * RHO_M, r_s=r_s, c=CONC)


def stage_ct():
    import cluster_toolkit as ct
    p = nfw_profile()
    k = np.logspace(-4, 3, 1200)
    xi_out = ct.xi.xi_mm_at_r(R3D, k, p.rho_tilde(k))
    res_rho = xi_out / p.rho(R3D) - 1.0

    r_xi = np.logspace(-3, 3, 1000)
    sig = ct.deltasigma.Sigma_at_R(R2D, r_xi, p.rho(r_xi) / RHO_M,
                                   M200M_H, CONC, OMEGA_M)
    res_sig = sig / (p.sigma(R2D) / 1e12) - 1.0

    ds = ct.deltasigma.DeltaSigma_at_R(R2D, R2D, p.sigma(R2D) / 1e12,
                                       M200M_H, CONC, OMEGA_M)
    res_ds = ds / (p.delta_sigma(R2D) / 1e12) - 1.0

    np.savez(os.path.join(OUT, "chain_res_ct.npz"), r3d=R3D, r2d=R2D,
             res_rho=res_rho, res_sig=res_sig, res_ds=res_ds)
    for lab, r in (("rho/xi", res_rho), ("Sigma", res_sig), ("DSigma", res_ds)):
        print(f"ct {lab:7s} max={np.max(np.abs(r)):.3e} "
              f"med={np.median(np.abs(r)):.3e}")


def stage_clenspy():
    from clenspy.utils.integrate import (compute_sigma_grid,
                                         pk_to_xi_fftlog,
                                         sigma_to_deltasigma_cumtrapz)
    import clmm
    from clmm.theory import (compute_3d_density, compute_surface_density,
                             compute_excess_surface_density)
    p = nfw_profile()

    # --- CLensPy transforms fed the closed forms -------------------------
    # k to 1e5, matching TwoHaloTerm's own internal _kfine window: FFTLog
    # ringing from a truncated k-window otherwise contaminates r < 0.3
    k = np.logspace(-4, 5, 2048)
    xi_out = np.asarray(pk_to_xi_fftlog(k, p.rho_tilde(k), R3D)).reshape(-1)
    res_rho_cl = xi_out / p.rho(R3D) - 1.0

    # precision bumped per review: 150 -> 600 Abel nodes
    sig = np.asarray(compute_sigma_grid(lambda r, z: p.rho(r), R2D,
                                        np.array([0.0]), method="trapz",
                                        rmax_integral=300.0,
                                        n_points=600)).reshape(-1)
    res_sig_cl = sig / p.sigma(R2D) - 1.0

    # precision bumped per review: deeper + denser interior grid
    r_ext = np.logspace(-4, np.log10(20.0), 1200)
    ds_ext = np.asarray(sigma_to_deltasigma_cumtrapz(
        r_ext, p.sigma(r_ext))).reshape(-1)
    ds = np.exp(np.interp(np.log(R2D), np.log(r_ext),
                          np.log(np.maximum(ds_ext, 1e-300))))
    res_ds_cl = ds / p.delta_sigma(R2D) - 1.0

    # --- CLMM native NFW (mean-density mass definition, z=0) -------------
    cosmo = clmm.Cosmology(H0=100.0 * H, Omega_dm0=OMEGA_M - OMEGA_B,
                           Omega_b0=OMEGA_B, Omega_k0=0.0)
    m_phys = M200M_H / H                      # Msun
    # clmm requires z_cl > 0; 1e-4 keeps comoving == physical (and the
    # 'mean' mass definition's (1+z)^3) to well below every code's own
    # accuracy floor
    kw = dict(mdelta=m_phys, cdelta=CONC, z_cl=1e-4, cosmo=cosmo,
              delta_mdef=200, halo_profile_model="nfw", massdef="mean")
    rho_clmm = np.asarray(compute_3d_density(R3D / H, **kw, verbose=False))
    res_rho_cm = (rho_clmm / H ** 2) / p.rho(R3D) - 1.0
    sig_clmm = np.asarray(compute_surface_density(R2D / H, **kw,
                                                  verbose=False))
    res_sig_cm = (sig_clmm / H) / p.sigma(R2D) - 1.0
    ds_clmm = np.asarray(compute_excess_surface_density(R2D / H, **kw,
                                                        verbose=False))
    res_ds_cm = (ds_clmm / H) / p.delta_sigma(R2D) - 1.0

    # --- CLMM backend (pyccl) fed the Fourier profile directly ----------
    # clmm's own API is parametric-only, but its pyccl backend accepts an
    # arbitrary Fourier profile: HaloProfile._fourier -> real / projected /
    # cumul2d via FFTLog. This is the generic P(k) -> Sigma / DeltaSigma
    # path through the CLMM stack.
    import pyccl

    class TildeProf(pyccl.halos.HaloProfile):
        def __init__(self, prof):
            super().__init__(mass_def=pyccl.halos.MassDef(200, "matter"))
            self.prof = prof

        def _fourier(self, cosmo, k, M, a):
            kk = np.atleast_1d(np.asarray(k, dtype=float))
            ft = self.prof.rho_tilde(kk)
            return ft if np.ndim(M) == 0 else np.tile(
                ft, (np.atleast_1d(M).size, 1))

    ccl_cosmo = pyccl.Cosmology(Omega_c=OMEGA_M - OMEGA_B, Omega_b=OMEGA_B,
                                h=H, sigma8=0.8238, n_s=0.9665)
    # FFTLog precision tuned per transform (scans in the session log):
    # real() prefers the wide window; projected/cumul2d the denser one
    tp_r = TildeProf(p)
    tp_r.update_precision_fftlog(padding_lo_fftlog=1e-4,
                                 padding_hi_fftlog=1e4,
                                 n_per_decade=600, plaw_fourier=-2.0)
    rho_pb = np.asarray(tp_r.real(ccl_cosmo, R3D, 1e14, 1.0))
    res_rho_pb = rho_pb / p.rho(R3D) - 1.0
    tp_s = TildeProf(p)
    tp_s.update_precision_fftlog(padding_lo_fftlog=1e-3,
                                 padding_hi_fftlog=1e3,
                                 n_per_decade=1200, plaw_fourier=-2.0)
    sig_pb = np.asarray(tp_s.projected(ccl_cosmo, R2D, 1e14, 1.0))
    res_sig_pb = sig_pb / p.sigma(R2D) - 1.0
    cum_pb = np.asarray(tp_s.cumul2d(ccl_cosmo, R2D, 1e14, 1.0))
    res_ds_pb = (cum_pb - sig_pb) / p.delta_sigma(R2D) - 1.0

    np.savez(os.path.join(OUT, "chain_res_clenspy.npz"), r3d=R3D, r2d=R2D,
             res_rho_cl=res_rho_cl, res_sig_cl=res_sig_cl,
             res_ds_cl=res_ds_cl, res_rho_cm=res_rho_cm,
             res_sig_cm=res_sig_cm, res_ds_cm=res_ds_cm,
             res_rho_pb=res_rho_pb, res_sig_pb=res_sig_pb,
             res_ds_pb=res_ds_pb)
    for lab, r in (("CLensPy rho", res_rho_cl), ("CLensPy Sig", res_sig_cl),
                   ("CLensPy DS", res_ds_cl), ("CLMM rho", res_rho_cm),
                   ("CLMM Sig", res_sig_cm), ("CLMM DS", res_ds_cm),
                   ("pycclFT rho", res_rho_pb), ("pycclFT Sig", res_sig_pb),
                   ("pycclFT DS", res_ds_pb)):
        print(f"{lab:12s} max={np.nanmax(np.abs(r)):.3e} "
              f"med={np.nanmedian(np.abs(r)):.3e}")


def stage_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    d_ct = np.load(os.path.join(OUT, "chain_res_ct.npz"))
    d_cl = np.load(os.path.join(OUT, "chain_res_clenspy.npz"))

    sns.set_theme(style="white", font_scale=1.8)
    C = {"cluster_toolkit": "black", "CLensPy": "firebrick",
         "CLMM": "grey", "CLMM backend ($P$ input)": "darkgrey"}
    LS = {"cluster_toolkit": "-", "CLensPy": "--", "CLMM": "-.",
          "CLMM backend ($P$ input)": ":"}

    panels = [
        (r"$\rho\;/\;\xi$ stage", d_ct["r3d"],
         [("cluster_toolkit", d_ct["res_rho"]),
          ("CLensPy", d_cl["res_rho_cl"]),
          ("CLMM", d_cl["res_rho_cm"]),
          ("CLMM backend ($P$ input)", d_cl["res_rho_pb"])],
         r"$r$ [cMpc/$h$]"),
        (r"$\Sigma(R)$ stage", d_ct["r2d"],
         [("cluster_toolkit", d_ct["res_sig"]),
          ("CLensPy", d_cl["res_sig_cl"]),
          ("CLMM", d_cl["res_sig_cm"]),
          ("CLMM backend ($P$ input)", d_cl["res_sig_pb"])],
         r"$R$ [cMpc/$h$]"),
        (r"$\Delta\Sigma(R)$ stage", d_ct["r2d"],
         [("cluster_toolkit", d_ct["res_ds"]),
          ("CLensPy", d_cl["res_ds_cl"]),
          ("CLMM", d_cl["res_ds_cm"]),
          ("CLMM backend ($P$ input)", d_cl["res_ds_pb"])],
         r"$R$ [cMpc/$h$]"),
    ]

    # one panel per stage; each carries an INSET zoom on the core
    # (r/R in [0.1, 1], LINEAR x with dense ticks, fixed +-0.01% scale,
    # label inside the box, connector lines to the zoomed region;
    # anything beyond +-0.01% -- CLMM's ~0.3% offset, ct's -0.03% Sigma
    # dip -- is deliberately off the inset scale)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    for ax, (title, x, curves, xlabel) in zip(axes, panels):
        for name, res in curves:
            ax.semilogx(x, 100.0 * res, color=C[name], ls=LS[name],
                        lw=3.0, label=name)
        ax.axhline(0, color="0.25", lw=0.8)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)

        axz = ax.inset_axes([0.56, 0.08, 0.40, 0.28])
        for name, res in curves:
            axz.plot(x, 100.0 * res, color=C[name], ls=LS[name], lw=2.0)
        axz.axhline(0, color="0.25", lw=0.6)
        axz.set_xlim(0.1, 1.0)
        axz.set_ylim(-0.01, 0.01)
        axz.set_xticks(np.arange(0.2, 1.01, 0.2))
        axz.set_xticks(np.arange(0.1, 1.01, 0.1), minor=True)
        axz.tick_params(axis="both", labelsize=11)
        axz.text(0.05, 0.82, r"core, $\pm 0.01\%$",
                 transform=axz.transAxes, fontsize=12)
        axz.set_facecolor("white")
        ax.indicate_inset_zoom(axz, edgecolor="0.4", lw=1.0)
    axes[0].set_ylabel("code / analytic $-$ 1  [%]")
    axes[0].legend(fontsize=14, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "chain_residuals_codes.pdf"))
    plt.close(fig)

    # residual summary table + macros
    rows = [
        ("cluster\\_toolkit", d_ct["res_rho"], d_ct["res_sig"], d_ct["res_ds"]),
        ("CLensPy", d_cl["res_rho_cl"], d_cl["res_sig_cl"], d_cl["res_ds_cl"]),
        ("CLMM (native NFW)", d_cl["res_rho_cm"], d_cl["res_sig_cm"],
         d_cl["res_ds_cm"]),
        ("CLMM backend ($P$ input)", d_cl["res_rho_pb"],
         d_cl["res_sig_pb"], d_cl["res_ds_pb"]),
    ]
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "chain_residuals_codes.tex"), "w") as f:
        f.write("% generated by 10_chain_residuals.py -- do not edit\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write(" & \\multicolumn{2}{c}{$\\rho/\\xi$}"
                " & \\multicolumn{2}{c}{$\\Sigma$}"
                " & \\multicolumn{2}{c}{$\\Delta\\Sigma$} \\\\\n")
        f.write("code & max & median & max & median & max & median \\\\\n")
        f.write("\\midrule\n")
        for name, a, b, c in rows:
            cells = []
            for r in (a, b, c):
                cells += [f"{np.nanmax(np.abs(r)):.2e}",
                          f"{np.nanmedian(np.abs(r)):.2e}"]
            f.write(name + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    keys = ["cluster_toolkit", "CLensPy", "CLMM", "CLMMFT"]
    # the pyccl-FFTLog rho residual grows in the far tail (r >> r200,
    # rho -> 0); quote its worst over the lensing-relevant r <= 5 range
    m5 = d_ct["r3d"] <= 5.0
    worst = {}
    for k, (_, a, b, c) in zip(keys, rows):
        a_eff = np.asarray(a)[m5] if k == "CLMMFT" else a
        worst[k] = max(np.nanmax(np.abs(a_eff)), np.nanmax(np.abs(b)),
                       np.nanmax(np.abs(c)))
    with open(os.path.join(HERE, "report",
                           "values_chain_residuals.tex"), "w") as f:
        f.write("% generated by 10_chain_residuals.py -- do not edit\n")
        f.write(f"\\newcommand{{\\resWorstCT}}"
                f"{{{worst['cluster_toolkit']:.1%}}}\n".replace("%", "\\%"))
        f.write(f"\\newcommand{{\\resWorstCLensPy}}"
                f"{{{worst['CLensPy']:.2%}}}\n".replace("%", "\\%"))
        f.write(f"\\newcommand{{\\resWorstCLMM}}"
                f"{{{worst['CLMM']:.2%}}}\n".replace("%", "\\%"))
        f.write(f"\\newcommand{{\\resWorstCLMMFT}}"
                f"{{{worst['CLMMFT']:.2%}}}\n".replace("%", "\\%"))
    print("worst per code:", {k: f"{v:.3%}" for k, v in worst.items()})
    print("wrote figure, table, macros")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["ct", "clenspy", "figure"])
    a = ap.parse_args()
    {"ct": stage_ct, "clenspy": stage_clenspy, "figure": stage_figure}[a.stage]()
