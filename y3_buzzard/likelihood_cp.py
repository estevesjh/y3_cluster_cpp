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
    """Build the summed tangential-shear theory vector.

    gamma_t^theory(R | i,j) = <gamma_t^1h>_i(R) + gamma_t^prj(R | i,j)

    shear1hsel/vals is N_i-weighted; divide entry-wise by the
    number-count integral (shape 12, broadcast across the R
    points per bin) to get the per-cluster average, then add the
    projection piece.
    """
    NC = np.asarray(block[config["num_counts_section"], "vals"]).ravel()
    S1h_Ni = np.asarray(block[config["shear_1h_section"], "vals"]).ravel()
    Sprj = np.asarray(block[config["shear_prj_section"], "vals"]).ravel()
    if NC.size != _NC_N_BINS:
        raise ValueError(
            f"likelihood_cp: numcountssel/vals size {NC.size} != "
            f"{_NC_N_BINS}")
    shear_n = _NC_N_BINS * config["shear_n_r"]
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
    NC_tile = np.repeat(NC, config["shear_n_r"])
    bad = NC_tile <= 0.0
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

    # Shear — theory = <gamma_t^1h> + gamma_t^prj.
    Shear_theory = _shear_theory(block, config)
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
