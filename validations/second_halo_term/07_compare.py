#!/usr/bin/env python
"""Compare everything, evaluate the gates, decide the DeltaSigma method,
and emit every report input (figures, LaTeX tables, values.tex macros).

Run with the conda pipeline env:
  /opt/homebrew/Caskroom/miniforge/base/envs/y3cl_je_macos/bin/python 07_compare.py

Inputs (outputs/*.npz from 01-06). Outputs to report/figs, report/tables,
report/values.tex. Nothing here is hand-edited; the LaTeX report includes
these files verbatim.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIGS = os.path.join(HERE, "report", "figs")
TABLES = os.path.join(HERE, "report", "tables")

# fixed identity colors (CVD-validated); linestyle = secondary encoding
C = {"truth": "#000000", "ct": "#0072B2", "clenspy": "#009E73",
     "pyccl": "#CC79A7", "clmm": "#E69F00", "prod": "#D55E00"}
LS = {"truth": "-", "ct": "--", "clenspy": "-.", "pyccl": ":",
      "clmm": (0, (3, 1, 1, 1)), "prod": "-"}

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})

R_PIN = np.array([3.0, 4.38, 6.41, 9.36, 13.69, 20.0])
IZ_SLICES = [0, 3, 5, 8]
GATES = []   # (name, value, threshold, passed)


def gate(name, value, thresh):
    ok = bool(value < thresh)
    GATES.append((name, value, thresh, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.3e} (< {thresh:g})")
    return ok


def loginterp(x, xp, fp):
    return np.interp(np.log(x), np.log(xp), fp)


def load(name):
    path = os.path.join(OUT, name)
    return np.load(path, allow_pickle=False) if os.path.exists(path) else None


def check_units(d, name):
    u = str(d["units_sigma"]) if "units_sigma" in d.files else None
    if u is not None and "Msun h/pc^2" not in u:
        raise SystemExit(f"unit ledger violation in {name}: units_sigma={u}")


def main():
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    pk = load("pk_camb.npz")
    cct = load("chain_ct.npz")
    ccl_ = load("chain_clenspy.npz")
    rct = load("ref_ct.npz")
    rcl = load("ref_clenspy.npz")
    prod = load("prod_before.npz")
    prod_after = {m: load(f"prod_after_{m}.npz") for m in ("sandwich", "direct")}
    for d, n in ((rct, "ref_ct"), (rcl, "ref_clenspy"), (prod, "prod_before")):
        check_units(d, n)

    z = pk["z"]
    r_sigma = rct["r_sigma"]
    vals = {}

    # ------------------------------------------------------------------ P(k)
    print("== P(k) gates ==")
    gate("pk_dump_vs_camb_max", float(pk["dump_dev_max"]), 0.02)
    vals["pkDumpDevMax"] = f"{float(pk['dump_dev_max']):.2%}"
    vals["pkDumpDevMedian"] = f"{float(pk['dump_dev_median']):.2%}"
    vals["sigmaEightAchieved"] = f"{float(pk['sigma8']):.4f}"

    # ------------------------------------------------------- chain bench gates
    print("== analytic chain bench gates ==")
    # P->xi: ct on smooth (untruncated NFW) and mcfit raw
    gate("chain_p2xi_ct_nfw_untrunc_max",
         float(cct["nfw_m14_c5__p2xi_untrunc_err"][0]), 0.01)
    gate("chain_p2xi_mcfit_gauss_s1_max",
         float(ccl_["gaussian_s1__p2xi_raw_err"][0]), 0.01)
    # xi->Sigma
    gate("chain_x2s_ct_worst_max",
         max(float(cct[f"{p}__x2s_err"][0]) for p in
             ("gaussian_s05", "gaussian_s1", "gaussian_s2",
              "exponential_h1", "nfw_m14_c5")), 0.01)
    gate("chain_x2s_clenspy_n150_worst_max",
         max(float(ccl_[f"{p}__x2s_n150_err"][0]) for p in
             ("gaussian_s05", "gaussian_s1", "gaussian_s2",
              "exponential_h1", "nfw_m14_c5")), 0.01)
    # Sigma->DS candidates (R>=0.5 fractional, worst profile)
    def s2d_worst(m):
        return max(float(cct[f"{p}__s2d_err_{m}_r05"][0]) for p in
                   ("gaussian_s05", "gaussian_s1", "gaussian_s2",
                    "exponential_h1", "nfw_m14_c5"))
    w_sand, w_dir = s2d_worst("sandwich"), s2d_worst("direct_extended")
    gate("chain_s2d_sandwich_worst_max", w_sand, 0.01)
    gate("chain_s2d_direct_extended_worst_max", w_dir, 0.01)
    vals["chainSandwichWorst"] = f"{w_sand:.1%}"
    vals["chainDirectWorst"] = f"{w_dir:.2%}"

    # ------------------------------------------------- fiducial xi 4-way gates
    print("== fiducial xi(r) 4-way ==")
    r_xi_cl = rcl["r_xi"]
    xi_rows = {}
    for iz in IZ_SLICES:
        xi_ct = loginterp(r_xi_cl, rct["rfix_dense"], rct[f"lin_iz{iz}_xi_dense"])
        xi_cl = rcl[f"lin_iz{iz}_xi"]
        xi_ccl = rcl[f"ccl_lin_iz{iz}_xi"]
        xi_rows[iz] = (xi_ct, xi_cl, xi_ccl)
    # mutual agreement away from the zero crossing: r in [0.5, 50]
    m = (r_xi_cl >= 0.5) & (r_xi_cl <= 50.0)
    worst_mutual = 0.0
    for iz in IZ_SLICES:
        a, b, c_ = xi_rows[iz]
        worst_mutual = max(worst_mutual,
                           np.nanmax(np.abs(b[m] / a[m] - 1)),
                           np.nanmax(np.abs(c_[m] / a[m] - 1)))
    gate("xi_mutual_ct_clenspy_pyccl_max", float(worst_mutual), 0.02)
    vals["xiMutualMax"] = f"{worst_mutual:.2%}"
    # z variation (references)
    ratio_ref = float(rct["lin_xi_r1_of_z"][0] / rct["lin_xi_r1_of_z"][-1])
    vals["xiZRatioRef"] = f"{ratio_ref:.1f}"
    # production Wp: z-degenerate before?
    wp = prod["camb_wp_hh"]
    vals["prodWpDegenerate"] = "yes" if bool(prod["camb_z_degenerate"]) else "no"

    # production xi (Wp row) vs ct reference at iz=5, before fix
    r_perp = prod["r_perp"]
    xi_ct5_on_perp = loginterp(r_perp, rct["rfix_dense"], rct["lin_iz5_xi_dense"])
    dev_wp_before = np.nanmax(np.abs(
        wp[5][r_perp >= 0.5] / xi_ct5_on_perp[r_perp >= 0.5] - 1))
    print(f"  production Wp(iz=5) vs ct xi (BEFORE): max dev {dev_wp_before:.1%}"
          f"  (z-degenerate: {vals['prodWpDegenerate']})")
    vals["wpDevBefore"] = f"{dev_wp_before:.0%}"

    # xi_nl dump table vs nl reference at slices (on its own r grid)
    xinl_r, xinl = prod["xinl_r"], prod["xinl_xi"]
    devs = []
    for iz in IZ_SLICES:
        ref = loginterp(xinl_r, rct["rfix_dense"], rct[f"nl_iz{iz}_xi_dense"])
        mm = (xinl_r >= 0.5) & (xinl_r <= 50.0)
        devs.append(np.nanmax(np.abs(xinl[iz][mm] / ref[mm] - 1)))
    vals["xiNlDevMax"] = f"{max(devs):.1%}"
    print(f"  dump xi_nl vs per-z nl reference: max dev {max(devs):.1%} "
          f"(recall: this fixture's 'nl' P(k) is the linear fallback -> "
          f"compare vs nl ref only if matter_power_nl existed; vs LIN:")
    devs_lin = []
    for iz in IZ_SLICES:
        ref = loginterp(xinl_r, rct["rfix_dense"], rct[f"lin_iz{iz}_xi_dense"])
        mm = (xinl_r >= 0.5) & (xinl_r <= 50.0)
        devs_lin.append(np.nanmax(np.abs(xinl[iz][mm] / ref[mm] - 1)))
    gate("xi_nl_dump_vs_lin_ref_max", float(max(devs_lin)), 0.02)

    # ---------------------------------- xi_nl linear-fallback physics impact
    # No nonlinear emulator exists (camb-emulator ships only camb_linear_*
    # models and no production ini sets nonlinear_pk_path), so
    # halo_model_cosmosis's fallback feeds LINEAR P(k) to the xi_nl table
    # every run: the shearPrj projection branch consumes linear xi under
    # the xi_nl name. Quantify what halofit would change.
    print("== xi_nl linear-fallback impact (halofit / linear) ==")
    xi_ratio_r1 = float(
        loginterp(1.0, rct["rfix_dense"], rct["nl_iz5_xi_dense"])
        / loginterp(1.0, rct["rfix_dense"], rct["lin_iz5_xi_dense"]))
    ds_ratio_r3 = float(
        loginterp(3.0, r_sigma, rct["nl_iz5_dsigma_anchor"])
        / loginterp(3.0, r_sigma, rct["lin_iz5_dsigma_anchor"]))
    print(f"  z=0.408: xi_nl/xi_lin(r=1) = {xi_ratio_r1:.2f}, "
          f"DSigma_2h nl/lin (R=3) = {ds_ratio_r3:.2f}")
    vals["xiNlLinRatioROne"] = f"{xi_ratio_r1:.2f}"
    vals["dsNlLinRatioRThree"] = f"{ds_ratio_r3:.2f}"

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(4.6, 4.6), sharex=True,
                                  height_ratios=[2, 1])
    rr = rct["rfix_dense"]
    mplot = (rr >= 0.1) & (rr <= 60.0)
    ax.loglog(rr[mplot], rct["lin_iz5_xi_dense"][mplot], C["ct"], ls=LS["ct"],
              lw=1.8, label="current: linear $P(k)$ (the silent fallback)")
    ax.loglog(rr[mplot], rct["nl_iz5_xi_dense"][mplot], C["prod"], lw=1.8,
              label="proposed fix: halofit $P(k)$")
    ax.set_ylabel(r"$\xi_{mm}(r,\,z=0.41)$  (dimensionless)")
    ax.legend(fontsize=7)
    axr.semilogx(rr[mplot],
                 rct["nl_iz5_xi_dense"][mplot] / rct["lin_iz5_xi_dense"][mplot],
                 C["prod"], lw=1.8)
    axr.axhline(1, color="k", lw=0.5)
    # focus on the observable-relevant range; the ratio reaches ~22 at
    # r = 0.1 (quoted in the caption), off-scale here by design
    axr.set_ylim(0.7, 6.0)
    axr.set_xlabel(r"$r$ [cMpc/$h$, comoving]")
    axr.set_ylabel("halofit / linear")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "xi_nl_linear_fallback.pdf"))
    plt.close(fig)

    # ------------------------------------------------ Sigma / DS: before state
    print("== production (BEFORE) vs references ==")
    sig_prod = prod["camb_sigma_hh"]
    sig_cl5 = rcl["lin_iz5_sigma"]
    dev_rows = []
    for rp in R_PIN:
        p_ = loginterp(rp, r_perp, sig_prod[5])
        c_ = loginterp(rp, r_sigma, sig_cl5)
        t_ = loginterp(rp, r_sigma, rct["lin_iz5_sigma"])
        dev_rows.append((rp, p_, c_, t_, p_ / c_ - 1))
    dev_max = max(abs(d[-1]) for d in dev_rows)
    print(f"  Sigma_hh(iz=5) vs CLensPy at pin radii: dev "
          f"{dev_rows[0][-1]:+.0%} .. {dev_rows[-1][-1]:+.0%}")
    vals["sigmaDevBeforeMin"] = f"{min(d[-1] for d in dev_rows):+.0%}"
    vals["sigmaDevBeforeMax"] = f"{max(d[-1] for d in dev_rows):+.0%}"
    vals["nanFracBefore"] = f"{int(prod['camb_nan_dsigma'])/sig_prod.size:.0%}"
    vals["nanEdgeR"] = f"{float(prod['camb_nan_edge_r']):.2f}"

    with open(os.path.join(TABLES, "deviation_before.tex"), "w") as f:
        f.write("% generated by 07_compare.py -- do not edit\n")
        f.write("\\begin{tabular}{rrrrr}\n\\toprule\n")
        f.write("$R$ [cMpc/$h$] & production $\\Sigma_{hh}$ & CLensPy & "
                "cluster\\_toolkit & deviation \\\\\n\\midrule\n")
        for rp, p_, c_, t_, d_ in dev_rows:
            f.write(f"{rp:.2f} & {p_:.3f} & {c_:.3f} & {t_:.3f} & {d_:+.0%} \\\\\n"
                    .replace("%", "\\%"))
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ------------------------------------------------- method decision (fiducial)
    print("== DeltaSigma method decision (fiducial, vs converged anchor) ==")
    dec_rows = []
    for iz in [3, 5, 8]:
        anchor = rct[f"lin_iz{iz}_dsigma_anchor"]
        m05 = r_sigma >= 0.5
        for meth in ("sandwich", "direct"):
            ds = rct[f"lin_iz{iz}_dsigma_{meth}"]
            dmax = float(np.nanmax(np.abs(ds[m05] / anchor[m05] - 1)))
            dmed = float(np.nanmedian(np.abs(ds[m05] / anchor[m05] - 1)))
            dec_rows.append((iz, meth, dmax, dmed))
        # CLensPy cross-validation of the anchor itself
        cl = rcl[f"lin_iz{iz}_dsigma"]
        xval = float(np.nanmax(np.abs(cl[m05] / anchor[m05] - 1)))
        print(f"  iz={iz}: anchor vs CLensPy DS max dev {xval:.2%}")
        if iz == 5:
            gate("dsigma_anchor_vs_clenspy_max", xval, 0.03)
            vals["anchorVsClenspy"] = f"{xval:.2%}"
    d_sand = max(r[2] for r in dec_rows if r[1] == "sandwich")
    d_dir = max(r[2] for r in dec_rows if r[1] == "direct")
    # cancellation residual at fiducial
    canc = 0.0
    for iz in [3, 5, 8]:
        a = rct[f"lin_iz{iz}_dsigma_sandwich"]
        b = rct[f"lin_iz{iz}_dsigma_sandwich_md13"]
        pk_scale = np.nanmax(np.abs(rct[f"lin_iz{iz}_dsigma_anchor"]))
        canc = max(canc, float(np.nanmax(
            np.abs((a - b)[r_sigma >= 0.5])) / pk_scale))
    print(f"  D_sandwich={d_sand:.2%}  D_direct={d_dir:.2%}  "
          f"cancellation residual={canc:.2%}")
    winner = "direct" if (d_dir < d_sand and abs(d_dir - d_sand) > 0.01) else \
             ("sandwich" if d_sand < 0.03 else "direct")
    # fold in the chain-bench first-principles gate: shipped method must be <1%
    if winner == "sandwich" and w_sand >= 0.01:
        winner = "direct"
    print(f"  --> WINNER: {winner}  (criterion: smaller D, <3%; "
          f"tie within 1pt -> sandwich; must pass <1% analytic gate)")
    vals["dSand"] = f"{d_sand:.2%}"
    vals["dDirect"] = f"{d_dir:.2%}"
    vals["cancResidual"] = f"{canc:.2%}"
    vals["methodWinner"] = winner
    gate("dsigma_winner_D", min(d_sand, d_dir), 0.03)

    with open(os.path.join(TABLES, "method_decision.tex"), "w") as f:
        f.write("% generated by 07_compare.py -- do not edit\n")
        f.write("\\begin{tabular}{clrr}\n\\toprule\n")
        f.write("$z$ & method & max dev & median dev \\\\\n\\midrule\n")
        for iz, meth, dmax, dmed in dec_rows:
            f.write(f"{z[iz]:.3f} & {meth} & {dmax:.2%} & {dmed:.2%} \\\\\n"
                    .replace("%", "\\%"))
        f.write("\\bottomrule\n\\end{tabular}\n")

    # chain-bench summary table
    with open(os.path.join(TABLES, "chain_bench.tex"), "w") as f:
        f.write("% generated by 07_compare.py -- do not edit\n")
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("profile & \\multicolumn{2}{c}{sandwich (max/med)} & "
                "\\multicolumn{2}{c}{direct (max/med)} \\\\\n\\midrule\n")
        for p in ("gaussian_s05", "gaussian_s1", "gaussian_s2",
                  "exponential_h1", "nfw_m14_c5"):
            s = cct[f"{p}__s2d_err_sandwich_r05"]
            d_ = cct[f"{p}__s2d_err_direct_extended_r05"]
            f.write(f"{p.replace('_', ' ')} & {s[0]:.1%} & {s[1]:.1%} & "
                    f"{d_[0]:.2%} & {d_[1]:.2%} \\\\\n".replace("%", "\\%"))
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ------------------------------------------------------------- post-fix gates
    for meth, pa in prod_after.items():
        if pa is None:
            continue
        print(f"== production (AFTER, {meth}) gates ==")
        sig_a = pa["camb_sigma_hh"]
        ds_a = pa["camb_dsigma_hh"]
        gate(f"after_{meth}_nan_count", float(np.sum(~np.isfinite(ds_a))), 1)
        gate(f"after_{meth}_z_still_degenerate",
             1.0 if bool(pa["camb_z_degenerate"]) else 0.0, 0.5)
        m1 = r_perp >= 1.0
        dev_s = np.nanmax(np.abs(
            loginterp(r_sigma[m1[:len(r_sigma)]] if False else r_sigma,
                      r_perp, sig_a[5]) / sig_cl5 - 1)[r_sigma >= 1.0])
        gate(f"after_{meth}_sigma_vs_clenspy_iz5", float(dev_s), 0.03)
        anchor5 = rct["lin_iz5_dsigma_anchor"]
        ds5 = loginterp(r_sigma, r_perp, ds_a[5])
        dev_d = np.nanmax(np.abs(ds5[r_sigma >= 1.0] / anchor5[r_sigma >= 1.0] - 1))
        gate(f"after_{meth}_dsigma_vs_anchor_iz5", float(dev_d), 0.05)

    # ------------------------------------------------------------------ figures
    make_figures(pk, cct, ccl_, rct, rcl, prod, prod_after, z, r_sigma,
                 r_perp, r_xi_cl, xi_rows)

    # ------------------------------------------------------------------- values
    with open(os.path.join(HERE, "report", "values.tex"), "w") as f:
        f.write("% generated by 07_compare.py -- do not edit\n")
        for k, v in vals.items():
            vv = str(v).replace("%", "\\%")
            f.write(f"\\newcommand{{\\{k}}}{{{vv}}}\n")

    print("== gate summary ==")
    n_fail = sum(1 for g in GATES if not g[3])
    for name, v, t, ok in GATES:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{len(GATES)-n_fail}/{len(GATES)} gates pass")


def make_figures(pk, cct, ccl_, rct, rcl, prod, prod_after, z, r_sigma,
                 r_perp, r_xi_cl, xi_rows):
    # -- fig: xi 4-way at iz=5 + ratio
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(4.6, 4.6), sharex=True,
                                  height_ratios=[2, 1])
    xi_ct, xi_cl, xi_ccl = xi_rows[5]
    ax.loglog(r_xi_cl, xi_ct, C["ct"], ls=LS["ct"],
              label="cluster_toolkit (Hankel quadrature)")
    ax.loglog(r_xi_cl, xi_cl, C["clenspy"], ls=LS["clenspy"],
              label="CLensPy (FFTLog)")
    ax.loglog(r_xi_cl, xi_ccl, C["pyccl"], ls=LS["pyccl"],
              label="pyccl (independent Boltzmann)")
    wp5 = prod["camb_wp_hh"][5]
    ax.loglog(r_perp, wp5, C["prod"], ls=LS["prod"], lw=1.8,
              label=r"Wp_hh table (current, $z$-degenerate)")
    xinl5 = loginterp(prod["xinl_r"], prod["xinl_r"], prod["xinl_xi"][5])
    ax.loglog(prod["xinl_r"], xinl5, C["clmm"], ls=LS["clmm"],
              label="xi_nl table (correct per-z)")
    ax.set_xlim(0.1, 60.0)
    ax.set_ylabel(r"$\xi_{mm}(r,\,z=0.41)$  (dimensionless)")
    ax.legend(fontsize=6.5)
    for arr, rr, key in ((xi_cl, r_xi_cl, "clenspy"), (xi_ccl, r_xi_cl, "pyccl")):
        axr.semilogx(rr, arr / xi_ct - 1, C[key], ls=LS[key])
    axr.semilogx(r_perp, wp5 / loginterp(r_perp, rct["rfix_dense"],
                                         rct["lin_iz5_xi_dense"]) - 1,
                 C["prod"], lw=1.8)
    axr.axhline(0, color="k", lw=0.5)
    axr.set_xlim(0.1, 60.0)
    axr.set_ylim(-0.1, 0.7)
    axr.set_xlabel(r"$r$ [cMpc/$h$, comoving]")
    axr.set_ylabel("fractional dev.\nto cluster_toolkit")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "xi_4way_z041.pdf"))
    plt.close(fig)

    # -- fig: xi z-variation at r=1
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(z, rct["lin_xi_r1_of_z"], C["ct"], ls=LS["ct"],
            label="reference (per-z P(k))")
    wp_r1 = np.array([loginterp(1.0, r_perp, prod["camb_wp_hh"][i])
                      for i in range(z.size)])
    ax.plot(z, wp_r1, C["prod"], lw=1.8, label="production Wp (before)")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\xi_{mm}(r=1\,\mathrm{cMpc}/h,\ z)$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "xi_z_variation.pdf"))
    plt.close(fig)

    # -- fig: Sigma_hh current vs refs at slices
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(4.6, 4.6), sharex=True,
                                  height_ratios=[2, 1])
    for iz, alpha in zip(IZ_SLICES, (0.35, 0.6, 1.0, 0.8)):
        ax.loglog(r_sigma, rcl[f"lin_iz{iz}_sigma"], C["clenspy"],
                  ls=LS["clenspy"], alpha=alpha,
                  label=rf"CLensPy $\Sigma_{{2h}}$, z={z[iz]:.2f}")
    ax.loglog(r_perp, prod["camb_sigma_hh"][5], C["prod"], lw=1.8,
              label="current (z-degenerate +\ndummy-NFW contaminated)")
    ax.set_ylabel(r"$\Sigma_{hh}$ [$M_\odot h/\mathrm{pc}^2$, comoving], $b=1$")
    ax.legend(fontsize=6.5)
    for iz, alpha in zip(IZ_SLICES, (0.35, 0.6, 1.0, 0.8)):
        axr.semilogx(r_sigma,
                     loginterp(r_sigma, r_perp, prod["camb_sigma_hh"][5])
                     / rcl[f"lin_iz{iz}_sigma"] - 1,
                     C["prod"], alpha=alpha)
    axr.axhline(0, color="k", lw=0.5)
    # focus on the documented +0.7..+1.7 deviation band; the small-R
    # dummy-NFW contamination (up to ~100x) is off-scale by design
    axr.set_ylim(-0.3, 2.0)
    axr.set_xlabel(r"$R$ [cMpc/$h$, comoving]")
    axr.set_ylabel("current / reference $-1$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "sigma_hh_before.pdf"))
    plt.close(fig)

    # -- fig: DeltaSigma methods at iz=5
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(4.6, 4.6), sharex=True,
                                  height_ratios=[2, 1])
    anchor = rct["lin_iz5_dsigma_anchor"]
    ax.loglog(r_sigma, np.abs(anchor), C["truth"], label="converged anchor")
    ax.loglog(r_sigma, np.abs(rct["lin_iz5_dsigma_sandwich"]), C["ct"],
              ls=LS["ct"], label="sandwich (repaired)")
    ax.loglog(r_sigma, np.abs(rct["lin_iz5_dsigma_direct"]), C["clenspy"],
              ls=LS["clenspy"], label="direct")
    ax.loglog(r_sigma, np.abs(rcl["lin_iz5_dsigma"]), C["pyccl"],
              ls=LS["pyccl"], label="CLensPy")
    if "clmm_iz5_dsigma_p2" in rcl.files:
        ax.loglog(r_sigma, np.abs(rcl["clmm_iz5_dsigma_p2"]), C["clmm"],
                  ls=LS["clmm"], label="clmm ($(1+z)^{-2}$)")
    ds_b = prod["camb_dsigma_hh"][5]
    ax.loglog(r_perp, np.abs(ds_b), C["prod"], lw=1.8,
              label="production (current, 60% NaN)")
    ax.set_ylabel(r"$|\Delta\Sigma_{hh}|$ [$M_\odot h/\mathrm{pc}^2$], $b=1$")
    ax.legend(fontsize=6.5)
    for key, col, ls in (("lin_iz5_dsigma_sandwich", C["ct"], LS["ct"]),
                         ("lin_iz5_dsigma_direct", C["clenspy"], LS["clenspy"])):
        axr.semilogx(r_sigma, rct[key] / anchor - 1, col, ls=ls)
    axr.semilogx(r_sigma, rcl["lin_iz5_dsigma"] / anchor - 1, C["pyccl"],
                 ls=LS["pyccl"])
    axr.axhline(0, color="k", lw=0.5)
    axr.set_ylim(-0.2, 0.2)
    axr.set_xlabel(r"$R$ [cMpc/$h$]")
    axr.set_ylabel("ratio $-1$ (to anchor)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "dsigma_methods_z041.pdf"))
    plt.close(fig)

    # -- fig: dSigma_hh row, current table vs fixed (replaces the old
    #    all-black NaN-mask imshow)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    row_cur = prod["dumptable_dsigma_hh"][5]
    finite = np.isfinite(row_cur)
    nan_edge = r_sigma[np.argmax(finite)]
    ax.axvspan(r_sigma[0], nan_edge, color="0.88", zorder=0)
    ax.text(np.sqrt(r_sigma[0] * nan_edge), 0.32,
            "NaN region\n(60% of table)", ha="center", fontsize=7,
            color="0.35")
    ax.loglog(r_sigma[finite], row_cur[finite], C["prod"], lw=1.8,
              label="current table (finite part,\nz-degenerate)")
    pa = prod_after.get("direct")
    if pa is not None:
        ax.loglog(r_perp, pa["camb_dsigma_hh"][5], C["clenspy"],
                  ls=LS["clenspy"], lw=1.8, label="fixed (direct), z=0.41")
    ax.loglog(r_sigma, np.abs(rct["lin_iz5_dsigma_anchor"]), C["truth"],
              lw=1.0, label="converged anchor, z=0.41")
    ax.set_xlabel(r"$R$ [cMpc/$h$, comoving]")
    ax.set_ylabel(r"$\Delta\Sigma_{hh}$ [$M_\odot h/\mathrm{pc}^2$], $b=1$")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "dsigma_row_current_fixed.pdf"))
    plt.close(fig)

    # -- fig: P(k) consistency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    k_h = pk["k_h"]
    for iz, alpha in zip(IZ_SLICES, (0.35, 0.55, 0.75, 1.0)):
        ax1.semilogx(k_h, pk["p_k_dump"][iz] / pk["p_k_lin"][iz] - 1,
                     C["ct"], alpha=alpha)
        ax2.loglog(k_h, pk["p_k_nl"][iz] / pk["p_k_lin"][iz], C["clenspy"],
                   alpha=alpha)
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel(r"$k$ [$h$/Mpc]")
    ax1.set_ylabel("dump / CAMB lin $-1$")
    ax2.set_xlabel(r"$k$ [$h$/Mpc]")
    ax2.set_ylabel(r"$P_{nl}/P_{lin}$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "pk_consistency.pdf"))
    plt.close(fig)

    # -- fig: chain bench (DS methods on gaussian_s1 + NFW)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=False)
    for ax, pname, title in zip(axes, ("gaussian_s1", "nfw_m14_c5"),
                                ("Gaussian $s=1$", "NFW $10^{14}, c=5$")):
        rs = cct["r_sigma"]
        truth = cct[f"{pname}__s2d_ds_true"]
        ax.semilogx(rs, cct[f"{pname}__s2d_ds_sandwich"] / truth - 1,
                    C["ct"], ls=LS["ct"], label="sandwich")
        ax.semilogx(rs, cct[f"{pname}__s2d_ds_direct_extended"] / truth - 1,
                    C["clenspy"], ls=LS["clenspy"], label="direct")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-0.4, 0.4)
        ax.set_xlabel(r"$R$ [cMpc/$h$]")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel(r"$\Delta\Sigma$ / closed form $-1$")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "chain_bench_methods.pdf"))
    plt.close(fig)

    # -- post-fix figures if available
    for meth, pa in prod_after.items():
        if pa is None:
            continue
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(4.6, 4.6), sharex=True,
                                      height_ratios=[2, 1])
        anchor = rct["lin_iz5_dsigma_anchor"]
        ax.loglog(r_sigma, np.abs(anchor), C["truth"], label="anchor")
        ds5 = pa["camb_dsigma_hh"][5]
        ax.loglog(r_perp, np.abs(ds5), C["prod"], lw=1.8,
                  label=f"fixed ({meth})")
        ax.loglog(r_sigma, np.abs(rcl["lin_iz5_dsigma"]), C["pyccl"],
                  ls=LS["pyccl"], label="CLensPy")
        ax.set_ylabel(r"$|\Delta\Sigma_{hh}|$")
        ax.legend(fontsize=7)
        axr.semilogx(r_sigma, loginterp(r_sigma, r_perp, ds5) / anchor - 1,
                     C["prod"], lw=1.8)
        axr.axhline(0, color="k", lw=0.5)
        axr.set_ylim(-0.05, 0.05)
        axr.set_xlabel(r"$R$ [cMpc/$h$, comoving]")
        axr.set_ylabel("fixed / anchor $-1$")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"dsigma_after_{meth}.pdf"))
        plt.close(fig)


if __name__ == "__main__":
    main()
