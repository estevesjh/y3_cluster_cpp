"""Gaussian likelihood for the mock_mcmc_cp_camb.ini production pipeline.

Compares fiducial (data, invcov) against two observables from the
Costanzi-2026 pipeline:

    Number counts         numcountssel/vals          -> 12
    Tangential shear      gamma_t^theory(R)          -> 180
        = <gamma_t^1h>(R) + gamma_t^prj(R)
        = (shear1hsel/vals / numcountssel/vals) + shear_prj/vals

The two shear contributions are added linearly — which is valid only
because we dropped the reduced-shear 1/(1 - sigma sci) denominator
(see src/modules/num_counts_sel/lensing_weights.hh and
src/models/sigma_prj_t.hh, both updated 2026-05-11).

shear1hsel/vals is the N_i-weighted integral of gamma_t^1h
(N_i[g]/N_i is the per-bin average); divide by numcountssel/vals to
get the bin-averaged 1-halo shear.  shear_prj/vals is already a
per-(lambda_bin, zob) shear ready to add in.

Diagonal invcov is the default (1-D array per observable); a dense
invcov (2-D array) is also accepted and used as a matmul.

Max-model mode (shear_max_section = shear1h2h_max in the ini section)
--------------------------------------------------------------------
Replaces the 1h + prj composition by the traditional max model
(Shear1h2hMax, docs/source/variants.md):

    gamma_t^theory(R) = shear1h2h_max/vals / numcounts/vals

no projection term. With is_b_proj_costanzi26 = T the max-model theory
is multiplied by the Costanzi-2026 selection-bias correction
(arXiv:2604.05833 App. C, src/pipelines/systematics/costanzi_bprj):

    gamma_t^theory(R) *= B_prj(R | lob_i, z_j),   R0 = R_lambda(lob) (1+z)

evaluated by costanzi_bprj.bprj_wall(block, R): A/alpha/beta/gamma
(values file) and the bin grid lob_centers / zob_centers (published by
the costanzi_bprj module stage) are read per sample from the datablock
section [costanzi_bprj]; only the radii come from this module's
shear_r_perp (comoving Mpc/h, the Shear1h2hMax r_perp grid). B_prj only
makes sense on the max model (the 1h + prj path already carries the
selection bias through b_sel), so the flag without shear_max_section is
a configuration error.

logL = -0.5 * sum_obs delta^T C^-1 delta, summed over NC and Shear.

Writes block["likelihoods", "likelihoods_like"].

Log-space option (log_space = T in the [likelihoods] ini section)
-----------------------------------------------------------------
Both NC and shear span >1 decade, so the linear-space Gaussian is a
poor model far from fiducial and the posterior is skewed/curved,
which slows MCMC mixing.  Working in y = ln(observable) flattens this.

We propagate the covariance with the delta method, linearized about
the *fixed* data vector d* = data (NOT per-sample theory):

    y = ln(d),  J = diag(1/d*)
    C_y       = J C J^T          ->  C_y[i,j]   = C[i,j] / (d*_i d*_j)
    C_y^{-1}                     ->  invcov_y[i,j] = d*_i d*_j invcov[i,j]

i.e. invcov_log = invcov * outer(d*, d*)   (dense)
                = invcov * d*^2            (diagonal)

Because d* is fixed, C_y is constant across the chain: the
normalization 1/2 ln det C_y and the change-of-variables Jacobian are
both sample-independent constants that drop out of MCMC.  The chi2 is

    chi2_log = (ln data - ln theory)^T invcov_log (ln data - ln theory)

At theory == data this is zero (maximal logL), so closure tests still
recover fiducial; to first order around fiducial chi2_log == chi2_lin,
so only the off-fiducial geometry changes.  invcov_log is precomputed
once in setup().
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from cosmosis.datablock import option_section

_NC_N_BINS = 12


def _chi2(delta: np.ndarray, invcov: np.ndarray) -> float:
    if invcov.ndim == 1:
        return float(np.sum(delta * delta * invcov))
    if invcov.ndim == 2:
        return float(delta @ invcov @ delta)
    raise ValueError(f"invcov ndim must be 1 or 2, got {invcov.ndim}")


def _to_log_space(d: np.ndarray, ic: np.ndarray) -> np.ndarray:
    """Delta-method invcov for y = ln(d), linearized about fixed d.

    invcov_y[i,j] = d_i d_j invcov[i,j]  (dense)
                  = d_i^2  invcov[i]      (diagonal)
    Requires d > 0 (true for NC counts and Delta-Sigma shear).
    """
    if np.any(d <= 0.0):
        raise ValueError("likelihood_cp: log_space requires data > 0; "
                         f"found min(data)={d.min():.3e}")
    if ic.ndim == 1:
        return ic * d * d
    return ic * np.outer(d, d)


def _costanzi_bprj_wall(config, r_grid):
    """Return wall(block) -> B_prj on the shear wall (Costanzi-2026, max model).

    Everything but the radii is read from the costanzi_bprj datablock
    section per sample (values-file parameters + the lob_centers /
    zob_centers the costanzi_bprj module publishes), see
    src/pipelines/systematics/costanzi_bprj/python/costanzi_bprj.py.
    """
    from systematics.costanzi_bprj.python.costanzi_bprj import bprj_wall
    
    if not config["shear_max_section"]:
        raise ValueError(
            "likelihood_cp: is_b_proj_costanzi26 requires "
            "shear_max_section (B_prj corrects the max model only)")
    if r_grid.size != config["shear_n_r"]:
        raise ValueError(
            f"likelihood_cp: shear_r_perp has {r_grid.size} radii but "
            f"data_Shear implies {config['shear_n_r']} per bin")
    pipelines_dir = str(Path(__file__).resolve().parents[1]
                        / "src" / "pipelines")
    if pipelines_dir not in sys.path:
        sys.path.insert(0, pipelines_dir)
    
    n_shear = _NC_N_BINS * config["shear_n_r"]

    def wall(block):
        B = bprj_wall(block, r_grid)
        if B.size != n_shear:
            raise ValueError(
                f"likelihood_cp: costanzi_bprj wall has {B.size} points, "
                f"shear theory has {n_shear}")
        return B

    return wall


def setup(options):
    fname = options[option_section, "filename"]
    vec = np.load(fname, allow_pickle=False)
    log_space = bool(options.get_bool(option_section, "log_space",
                                      default=False))
    data_shear = np.asarray(vec["data_Shear"]).ravel()
    if data_shear.size % _NC_N_BINS:
        raise ValueError(
            "likelihood_cp: data_Shear size must be a multiple of "
            f"{_NC_N_BINS}, got {data_shear.size}")
    config = {"filename": fname,
              "log_space": log_space,
              "num_counts_section": options.get_string(
                  option_section, "num_counts_section",
                  default="numcountssel"),
              "shear_1h_section": options.get_string(
                  option_section, "shear_1h_section",
                  default="shear1hmissel"),
              "shear_prj_section": options.get_string(
                  option_section, "shear_prj_section",
                  default="shear_prj"),
              "shear_n_r": data_shear.size // _NC_N_BINS,
              "verbose": bool(options.get_bool(option_section, "verbose",
                                                default=False))}
    # Max-model mode: theory = shear_max_section/vals / NC, no projection
    # term (see module docstring). Empty (default) keeps 1h + prj.
    config["shear_max_section"] = options.get_string(
        option_section, "shear_max_section", default="")

    # Optional shear scale cut (the Buzzard scale-split runs): keep only
    # radii with shear_r_min <= r_perp <= shear_r_max [cMpc/h]. The full
    # shear vector is loaded and validated first, then cut; the dense
    # inverse covariance is cut by COV-slicing (invert -> slice ->
    # invert), i.e. the removed radii are marginalized, not conditioned.
    # Defaults keep everything. NOTE: this option previously existed
    # only as an uncommitted NERSC working-tree patch -- the committed
    # scale-split run scripts silently ignored it on a fresh checkout
    # (des-nersc-cluster-scripts#3); this is the reproducible version.
    shear_r_min = float(options.get_double(option_section, "shear_r_min",
                                           default=0.0))
    shear_r_max = float(options.get_double(option_section, "shear_r_max",
                                           default=np.inf))
    r_grid = np.array([float(x) for x in options.get_string(
        option_section, "shear_r_perp",
        default="0.20000 0.28599 0.40896 0.58480 0.83625 "
                "1.19581 1.70998 2.44521 3.49658 5.00000").split()])
    keep_r = (r_grid >= shear_r_min) & (r_grid <= shear_r_max)
    config["shear_cut_active"] = not keep_r.all()
    if config["shear_cut_active"] and config["shear_n_r"] != r_grid.size:
        raise ValueError(
            "likelihood_cp: shear scale cut assumes the "
            f"{r_grid.size}-point r_perp grid, but data_Shear implies "
            f"{config['shear_n_r']} radii per bin")
    config["shear_mask"] = np.tile(keep_r, _NC_N_BINS)
    if config["shear_cut_active"] and not keep_r.any():
        raise ValueError("likelihood_cp: shear scale cut removes every "
                         f"radius ({shear_r_min}, {shear_r_max})")

    # rho_m(z) density-evolution factor on the shear theory: multiply the
    # per-bin DeltaSigma by (1+z_bin)^shear_1pz_power. The 1-halo term is
    # built with the COMOVING rho_m0 (frozen z=0, halo_model_cosmosis.py),
    # so a comoving-vs-physical surface-density mismatch with the Buzzard DV
    # shows up as a coherent z-tilt (data/theory grows with z); a single
    # power of (1+z) absorbs it (chi2 121080->30862 on the Buzzard DV).
    # This lives here, not in halo_model, because halo_model publishes one
    # z-agnostic 1-halo table consumed by all z-bins -- only the likelihood
    # sees the per-bin redshift. Default power 0.0 => factor 1 (no change).
    # Bins are z-major: index = z*4 + lambda; shear index = bin*shear_n_r+r.
    z_power = float(options.get_double(option_section, "shear_1pz_power",
                                      default=0.0))
    try:
        zreps_str = options.get_string(option_section, "shear_zbin_reps",
                                       default="0.275 0.435 0.575")
    except Exception:
        zreps_str = "0.275 0.435 0.575"
    if z_power != 0.0:
        zreps = np.array([float(x) for x in zreps_str.split()])
        if zreps.size != _NC_N_BINS // 4:
            raise ValueError("likelihood_cp: shear_zbin_reps needs "
                             f"{_NC_N_BINS // 4} redshifts, got {zreps.size}")
        fac_bin = (1.0 + np.repeat(zreps, 4)) ** z_power      # (12,)
        config["shear_1pz_factor"] = np.repeat(
            fac_bin, config["shear_n_r"])                     # (shear,)
        print(f"[likelihood_cp] shear rho_m(z) factor (1+z)^{z_power} "
              f"with z_bins={zreps.tolist()}")
    else:
        config["shear_1pz_factor"] = np.ones(data_shear.size)

    # Costanzi-2026 B_prj(R) on the max model (module docstring).
    config["is_b_proj_costanzi26"] = bool(options.get_bool(
        option_section, "is_b_proj_costanzi26", default=False))
    if config["is_b_proj_costanzi26"]:
        config["bprj_wall"] = _costanzi_bprj_wall(config, r_grid)
    expected = [("NC", _NC_N_BINS), ("Shear", data_shear.size)]
    for name, expected_n in expected:
        d = np.asarray(vec[f"data_{name}"]).ravel()
        ic = np.asarray(vec[f"invcov_{name}"])
        if d.size != expected_n:
            raise ValueError(
                f"likelihood_cp: data_{name} has size {d.size}, "
                f"expected {expected_n}")
        if ic.ndim == 1 and ic.size != expected_n:
            raise ValueError(
                f"likelihood_cp: invcov_{name} diagonal has size {ic.size}, "
                f"expected {expected_n}")
        if ic.ndim == 2 and ic.shape != (expected_n, expected_n):
            raise ValueError(
                f"likelihood_cp: invcov_{name} dense has shape {ic.shape}, "
                f"expected ({expected_n},{expected_n})")
        if name == "Shear" and config["shear_cut_active"]:
            m = config["shear_mask"]
            d = d[m]
            if ic.ndim == 1:
                ic = ic[m]
            else:
                cov = np.linalg.inv(ic)
                ic = np.linalg.inv(cov[np.ix_(m, m)])
        if log_space:
            # store ln(data) and the delta-method invcov; both fixed
            # across the chain (linearized about the data, not theory).
            config[f"data_{name}"] = np.log(d)
            config[f"invcov_{name}"] = _to_log_space(d, ic)
        else:
            config[f"data_{name}"] = d
            config[f"invcov_{name}"] = ic
    print(f"[likelihood_cp] loaded mock DV from {fname}: "
          f"NC={config['data_NC'].size}, Shear={config['data_Shear'].size}, "
          f"log_space={log_space}")
    return config


def _shear_theory(block, config) -> np.ndarray:
    """Build the tangential-shear theory vector.

    Default:   gamma_t^theory(R | i,j) = <gamma_t^1h>_i(R) + gamma_t^prj(R | i,j)
    Max model: gamma_t^theory(R | i,j) = <gamma_t^max>_i(R)   (shear_max_section)

    shear1hsel/vals and shear1h2h_max/vals are N_i-weighted; divide
    entry-wise by the number-count integral (shape 12, broadcast across
    the R points per bin) to get the per-cluster average, then (default
    only) add the projection piece.
    """
    NC = np.asarray(block[config["num_counts_section"], "vals"]).ravel()
    if NC.size != _NC_N_BINS:
        raise ValueError(
            f"likelihood_cp: numcountssel/vals size {NC.size} != "
            f"{_NC_N_BINS}")
    shear_n = _NC_N_BINS * config["shear_n_r"]
    NC_tile = np.repeat(NC, config["shear_n_r"])
    bad = NC_tile <= 0.0
    if config["shear_max_section"]:
        Smax_Ni = np.asarray(
            block[config["shear_max_section"], "vals"]).ravel()
        if Smax_Ni.size != shear_n:
            raise ValueError(
                f"likelihood_cp: {config['shear_max_section']}/vals size "
                f"{Smax_Ni.size} != {shear_n}")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(bad, 0.0, Smax_Ni / NC_tile)
    S1h_Ni = np.asarray(block[config["shear_1h_section"], "vals"]).ravel()
    # DeltaSigma_prj: use the CLUSTERED component only. shear_prj/vals = rnd + cl,
    # but for a *differential* DeltaSigma the mean-field (rnd) term must cancel to
    # zero -- verified: cl converges to b_large*rho_m*xi (=shear1h2hMax) at large R
    # while total (rnd+cl) diverges (RichnessSelection#1). Fall back to vals if the
    # module doesn't publish the rnd/cl split.
    prj_sec = config["shear_prj_section"]
    if block.has_value(prj_sec, "cl"):
        Sprj = np.asarray(block[prj_sec, "cl"]).ravel()
    else:
        Sprj = np.asarray(block[prj_sec, "vals"]).ravel()
    if S1h_Ni.size != shear_n:
        raise ValueError(
            f"likelihood_cp: shear1h/vals size {S1h_Ni.size} != "
            f"{shear_n}")
    if Sprj.size != shear_n:
        raise ValueError(
            f"likelihood_cp: shear_prj/vals size {Sprj.size} != "
            f"{shear_n}")
    # shear1hsel wall = (bin_index fast, r_perp slow) for Cartesian
    # product; NumCountsSel wall is just bin_index.  The two module
    # outputs share the same 12-bin ordering, so we can tile NC across
    # the configured R-per-bin axis.
    with np.errstate(divide="ignore", invalid="ignore"):
        S1h_avg = np.where(bad, 0.0, S1h_Ni / NC_tile)
    return S1h_avg + Sprj


def _residual(data, theory, log_space):
    """delta = data - theory, in ln-space if requested.

    In log_space, config[data_*] already holds ln(data); take ln of the
    theory here.  theory is clipped at a tiny floor so a transient
    non-positive prediction can't blow up the log (it just produces a
    large finite chi2 that the sampler rejects).
    """
    if not log_space:
        return data - theory
    return data - np.log(np.maximum(theory, 1e-300))


def execute(block, config):
    logL = 0.0
    parts = {}
    log_space = config["log_space"]

    # NumCounts — direct Gaussian on the 12-bin vector.
    NC_theory = np.asarray(
        block[config["num_counts_section"], "vals"]).ravel()
    delta_NC = _residual(config["data_NC"], NC_theory, log_space)
    parts["NC"] = -0.5 * _chi2(delta_NC, config["invcov_NC"])
    logL += parts["NC"]

    # Shear — theory = <gamma_t^1h> + gamma_t^prj,
    # scale-cut to match the data when shear_r_min/max is set.
    Shear_theory = _shear_theory(block, config)
    if config["is_b_proj_costanzi26"]:
        Shear_theory = Shear_theory * config["bprj_wall"](block)
    Shear_theory = Shear_theory * config["shear_1pz_factor"]   # rho_m(z) (1+z)^p
    if config["shear_cut_active"]:
        Shear_theory = Shear_theory[config["shear_mask"]]
    delta_Shear = _residual(config["data_Shear"], Shear_theory, log_space)
    parts["Shear"] = -0.5 * _chi2(delta_Shear, config["invcov_Shear"])
    logL += parts["Shear"]

    if config["verbose"]:
        print(f"[likelihood_cp] logL={logL:.4e}  "
              f"(NC={parts['NC']:.3e}, Shear={parts['Shear']:.3e})")

    block["likelihoods", "likelihoods_like"] = float(logL)
    return 0


def cleanup(config):
    return 0
