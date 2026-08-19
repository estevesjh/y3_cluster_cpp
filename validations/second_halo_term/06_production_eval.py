#!/usr/bin/env python
"""Drive the PRODUCTION two-halo code path exactly as halo_model_cosmosis
does, on the shared CAMB P(k) and on the dump's own P(k).

Run with the conda pipeline env:
  /opt/homebrew/Caskroom/miniforge/base/envs/y3cl_je_macos/bin/python 06_production_eval.py --tag before

Records Sigma_hh / dSigma_hh / Wp (xi_2halo) tables, NaN statistics, and
z-degeneracy diagnostics; also snapshots the checked-in dump's published
tables (halomodel/{sigma,dsigma,wp}_hh, xi_nl) for the 4-way xi comparison.

--tag before : the unfixed code (baseline; run once, output committed)
--tag after  : post-fix; add --method sandwich|direct if the fixed
               ct_2hTerm exposes dsigma_method.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DUMP = os.path.join(REPO, "docs", "figs", "real_pipeline_extract_prj2h_output")
sys.path.insert(0, os.path.join(REPO, "y3_buzzard"))

OMEGA_M = 0.311049


def run_production(r_perp, z, k_h, p_k, method=None):
    import haloModel as hm

    lens = hm.lensingModel(r_perp, omega_m=OMEGA_M, odelta=200)
    if method is not None:
        # post-fix path: rebuild the producer with an explicit method,
        # mirroring second_halo_term's call site
        p2h = hm.ct_2hTerm(OMEGA_M, Md=1e14, cd=5, bias=1.0,
                           dsigma_method=method)
        p2h.pk_to_dsigma(r_perp, k_h, p_k, z)
        return p2h.Sigma, p2h.dSigma, p2h.Xi
    lens.second_halo_term(z, k_h, p_k)
    return lens.Sigma["2h"], lens.dSigma["2h"], lens.Wp


def table_stats(name, arr):
    arr = np.asarray(arr)
    n_nan = int(np.sum(~np.isfinite(arr)))
    rows_identical = bool(all(np.array_equal(arr[0], arr[i])
                              for i in range(arr.shape[0])))
    print(f"  {name}: shape={arr.shape} non-finite={n_nan}/{arr.size} "
          f"({n_nan/arr.size:.1%}) all-z-rows-identical={rows_identical}")
    return n_nan, rows_identical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="before")
    ap.add_argument("--method", default=None,
                    help="post-fix: dsigma_method to evaluate")
    args = ap.parse_args()

    d = np.load(os.path.join(HERE, "outputs", "pk_camb.npz"), allow_pickle=False)
    k_h, z = d["k_h"], d["z"]
    r_perp = np.loadtxt(os.path.join(DUMP, "halomodel", "r_sigma.txt"))

    results = {"r_perp": r_perp, "z": z}

    for pk_name, ptag in (("p_k_lin", "camb"), ("p_k_dump", "dump")):
        print(f"production ct_2hTerm on {ptag} P(k):")
        sig, dsig, wp = run_production(r_perp, z, k_h, d[pk_name],
                                       method=args.method)
        results[f"{ptag}_sigma_hh"] = np.asarray(sig)
        results[f"{ptag}_dsigma_hh"] = np.asarray(dsig)
        results[f"{ptag}_wp_hh"] = np.asarray(wp)
        n_nan_s, ident_s = table_stats("Sigma_hh", sig)
        n_nan_d, ident_d = table_stats("dSigma_hh", dsig)
        table_stats("Wp (xi_2halo)", wp)
        results[f"{ptag}_nan_dsigma"] = np.int64(n_nan_d)
        results[f"{ptag}_z_degenerate"] = np.bool_(ident_s)
        # NaN edge radius (first finite column of a mid row)
        dsig_arr = np.asarray(dsig)
        finite_cols = np.isfinite(dsig_arr[5])
        edge = r_perp[np.argmax(finite_cols)] if finite_cols.any() else np.nan
        print(f"  dSigma_hh first finite column at R = {edge:.3f} (row iz=5)")
        results[f"{ptag}_nan_edge_r"] = np.float64(edge)

    # snapshot of the checked-in dump's published tables
    hm_dir = os.path.join(DUMP, "halomodel")
    for key in ("sigma_hh", "dsigma_hh", "wp_hh"):
        t = np.loadtxt(os.path.join(hm_dir, f"{key}.txt"))
        results[f"dumptable_{key}"] = t
    results["dumptable_r_sigma"] = np.loadtxt(os.path.join(hm_dir, "r_sigma.txt"))
    results["dumptable_z"] = np.loadtxt(os.path.join(hm_dir, "z.txt"))
    xi_dir = os.path.join(DUMP, "xi_nl")
    results["xinl_r"] = np.loadtxt(os.path.join(xi_dir, "r.txt"))
    results["xinl_z"] = np.loadtxt(os.path.join(xi_dir, "z.txt"))
    results["xinl_xi"] = np.loadtxt(os.path.join(xi_dir, "xi_nl.txt"))
    print(f"dump tables snapshotted: xi_nl {results['xinl_xi'].shape}, "
          f"dsigma_hh {results['dumptable_dsigma_hh'].shape}")

    results["units_r"] = "Mpc/h comoving"
    results["units_sigma"] = "Msun h/pc^2 comoving, b=1"
    results["tag"] = args.tag
    suffix = args.tag + (f"_{args.method}" if args.method else "")
    out_path = os.path.join(HERE, "outputs", f"prod_{suffix}.npz")
    np.savez(out_path, **results)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
