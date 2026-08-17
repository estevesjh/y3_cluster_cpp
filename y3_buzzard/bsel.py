"""Assemble the marginalised selection bias from P1/I1/I2 CPP outputs.

This is the Python closing stage of the Costanzi-2026 b_sel pipeline.
Reads the three datablock sections written by the CPP scalar integrands
(b_sel_marg_P1 / b_sel_marg_I1 / b_sel_marg_I2).

Datablock reads:
    b_sel_marg_P1/vals       -- shape (1, N_zob * N_lambda_bin)
    b_sel_marg_I1/vals       -- same shape
    b_sel_marg_I2/vals       -- same shape
    b_sel_marg_P1/zo_low     -- length (N_zob * N_lambda_bin)
    b_sel_marg_P1/zo_high    -- same length
    b_sel_marg_P1/lambda_bin -- same length (integer bin indices 0..3)
    cluster_mor/*            -- HOD scalars (if we need them for the
                                MOR prior in the ltr marginalisation)

Datablock writes (section b_sel_marginalised):
    lob        -- (N_lob,)    bin centres
    zob        -- (N_zob,)    bin centres
    theta      -- (N_theta,)  angular separations [rad]
    vals       -- (N_lob, N_zob, N_theta)  marginalised b_sel

Closed-form assembly (sel_bias.py:329-372):
    Delta_RND(lob, zob) = P1 + b_eff * I2
    delta_prj(lob, zob, ltr) = (lob - ltr) / Delta_RND - 1
    b_infty = b_eff * (1 + 0.13 * delta_prj)
    b_zero  = ((lob - ltr) - P1 - b_infty * I1) / (I2 - I1)
    sigma(theta | theta_lob) = 1 / (1 + exp(-k * (theta - theta0)))
       with theta_lob = R_lambda(lob) * (1+zob) / chi(zob),
            k = 2.5 / theta_lob, theta0 = theta_lob / 2
    b_sel(theta | lob, zob, ltr) = b_zero + (b_infty - b_zero) * sigma(theta)

Marginalisation over ltr is a single GL quadrature with weight
P(lob | ltr, zob) * prior(ltr, zob), exactly like the Python
SelBias.b_sel_marginalised.
"""
import time

import numpy as np
from cosmosis.datablock import option_section
from scipy.special import erfc, gammaln

from y3_buzzard.prj_params import PrjParams


# -- fixed bin-centre tables (match PROJ_Y3_B_i_t bin_edges in C++) --------
Y3_LAM_EDGES  = np.array([20.0, 30.0, 45.0, 60.0, 200.0])   # last edge capped
Y3_LAM_CENTRES = 0.5 * (Y3_LAM_EDGES[:-1] + Y3_LAM_EDGES[1:])
N_Y3_LAM_BIN   = 4


# -- closed-form helpers --------------------------------------------------

def _R_lambda(lam):
    return (lam / 100.0) ** 0.2


def _theta_lob(lob, zob, chi_of_z):
    return _R_lambda(lob) * (1.0 + zob) / chi_of_z(zob)


def _sigmoid(theta, theta_lob, damping=2.5, theta0_frac=0.5):
    k = damping / theta_lob
    theta0 = theta0_frac * theta_lob
    return 1.0 / (1.0 + np.exp(-k * (theta - theta0)))


def _b_eff(P1, I2):
    # b_eff is actually a halo-mass-weighted bias independent of projection.
    # Here we approximate b_eff via Delta_RND / I2 only if we got it; else
    # we store it from the CPP step. For now, a simple stand-in: b_eff ~
    # ratio (Delta_RND - P1) / I2. CPP integrand returns b(M,z) in I2 with
    # the correct weight so this recovers the original b_eff definition.
    # (TODO: if we add a b_eff CPP scalar explicitly, drop this stand-in.)
    raise NotImplementedError


# -- P(lob | ltr, z) EMG kernel (from plob_ltr C++ equivalent) -----------

# The C++ side uses lc_lt_t.hh with frozen Y3 tables baked in. The Python
# RichnessSelection package uses plob_ltr.py. For the timing / ratio test
# we reimplement the 5-parameter EMG here; caller can swap for the real
# lc_lt table via an `options` path. Shares parameters with MOR_HOD_t.

def _p_lob_given_ltr_emg(lob, ltr, z, mu, sigma, tau, fprj):
    """Full EMG: (1-fprj) N(mu, sigma) + fprj (tau/2) e^(...) erfc(...)."""
    u = (lob - mu) / sigma
    gauss = np.exp(-0.5 * u * u) / (sigma * np.sqrt(2.0 * np.pi))
    arg_exp  = 0.5 * tau * (2.0 * mu + tau * sigma ** 2 - 2.0 * lob)
    arg_erfc = (mu + tau * sigma ** 2 - lob) / (np.sqrt(2.0) * sigma)
    emg = 0.5 * tau * np.exp(arg_exp) * erfc(arg_erfc)
    return (1.0 - fprj) * gauss + fprj * emg


# -- MOR P(ltr | M, z) from cluster_mor scalars (HOD continuous-Poisson) --

def _l_sat(M, z, mor):
    """Shifted-Poisson satellite richness l_sat(M, z) (matches MOR.l_sat
    in richness_selection.mor)."""
    Mmin = 10.0 ** mor["log10_Mmin"]
    M1   = 10.0 ** mor["log10_M1"]
    alpha   = mor["alpha"]
    epsilon = mor["epsilon"]
    z_pivot = mor.get("z_pivot", 0.4544)
    dM1 = M1 - Mmin
    M = np.asarray(M, dtype=float)
    dM = np.maximum(M - Mmin, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.clip(dM / dM1, 1e-30, None)
        return frac ** alpha * ((1.0 + z) / (1.0 + z_pivot)) ** epsilon


def _p_ltr_given_M(ltr, M, z, mor):
    """P(ltr | M, z) via Poisson-Gaussian convolution.

    Shape: broadcasts over (ltr, M).  Matches richness_selection.mor.MOR.pdf
    exactly (scipy.special.gammaln for the log-factorial rather than the
    deprecated np.math.gamma used in the old version of this helper).
    """
    sigma_intr = mor["sigma_lambda"]    # the C++ ini calls it sigma_lambda
    m  = _l_sat(M, z, mor)
    mi = (m * sigma_intr) ** 2
    lam = m + mi
    x   = ltr + mi
    with np.errstate(divide="ignore", invalid="ignore"):
        log_val = (-lam
                   + (x - 1.0) * np.log(np.clip(lam, 1e-300, None))
                   - gammaln(np.clip(x, 1e-30, None)))
    val = np.exp(log_val)
    return np.where(np.asarray(ltr) >= 0.0, val, 0.0)


# -- setup / execute ------------------------------------------------------

def setup(options):
    n_theta  = options.get_int(option_section, "n_theta",  default=32)
    theta_lo = options.get_double(option_section, "theta_lo", default=1e-4)
    theta_hi = options.get_double(option_section, "theta_hi", default=5e-3)
    theta    = np.geomspace(theta_lo, theta_hi, n_theta)

    # Per-lob ltr integration range [ltr_lo, ltr_hi_factor * lob], GL
    # nodes built per-lob in execute() since the upper edge depends on
    # the bin centre.  Matches the Python SelBias.b_sel_marginalised
    # recipe which uses [1, 3*lob].
    ltr_lo          = options.get_double(option_section, "ltr_lo",          default=1.0)
    ltr_hi_factor   = options.get_double(option_section, "ltr_hi_factor",   default=3.0)
    n_ltr           = options.get_int(option_section,    "n_ltr",           default=128)
    # legacy fixed-range override (ltr_hi > 0 activates it)
    ltr_hi_fixed    = options.get_double(option_section, "ltr_hi",          default=0.0)

    # Wall-grid axes are passed explicitly via ini (not echoed back
    # by the CPP scalar-integration module).
    def _doubles(name, default):
        try:
            return np.asarray(options.get_double_array_1d(option_section, name),
                              dtype=float)
        except Exception:
            try:
                return np.asarray(options.get_int_array_1d(option_section, name),
                                  dtype=float)
            except Exception:
                return np.asarray(default, dtype=float)

    zob = _doubles("zob", [0.275, 0.425, 0.575])   # bin centres
    lob = _doubles("lob", Y3_LAM_CENTRES)

    # b_eff integration range; matches richness_selection.sel_bias defaults
    min_mass4integral = options.get_double(option_section,
                                           "min_mass4integral",
                                           default=1.0e13)
    ln_M_max_log10    = options.get_double(option_section,
                                           "ln_M_max_log10",
                                           default=15.5)
    n_m_beff          = options.get_int(option_section, "n_m_beff",
                                        default=100)

    # plob_ltr EMG coefficients come from the in-code PrjParams table.
    plob_params = PrjParams.default().as_dict()

    # Verbose mode: per-sample timing line + bias-asymptote tables.
    # Off by default so MCMC chains do not flood stdout; set
    # [bsel] verbose = T (ini) to re-enable for smoke runs.
    verbose = options.get_bool(option_section, "verbose", default=False)

    if verbose:
        print("[bsel.py] plob_ltr_params: PrjParams.default()", flush=True)

    return dict(
        theta=theta, zob=zob, lob=lob,
        ltr_lo=ltr_lo,
        ltr_hi_factor=ltr_hi_factor,
        ltr_hi_fixed=ltr_hi_fixed,
        n_ltr=n_ltr,
        min_mass4integral=min_mass4integral,
        ln_M_max_log10=ln_M_max_log10,
        n_m_beff=n_m_beff,
        plob_params=plob_params,
        verbose=verbose,
    )


def _read_vec_1d(block, section, key):
    """Try float array; fall back to int array (cosmosis distinguishes)."""
    try:
        return np.asarray(block.get_double_array_1d(section, key), dtype=float)
    except Exception:
        try:
            return np.asarray(block.get_int_array_1d(section, key),
                              dtype=float)
        except Exception:
            return np.asarray(block[section, key], dtype=float).ravel()


def _compute_b_eff(lob_i, zob_j, mor, M_grid, n_M, b_M):
    """b_eff(lob, zob) = int dM n(M,z) P(ltr=lob|M,z) M b(M,z)
                        / int dM n(M,z) P(ltr=lob|M,z) M

    Matches richness_selection.sel_bias.SelBias.b_eff (sel_bias.py:93).
    ``n_M`` and ``b_M`` are evaluated at ``zob_j`` on the ``M_grid``
    (log-spaced).  Integration is trapezoidal in ``ln M`` (same as the
    Python reference)."""
    P = _p_ltr_given_M(lob_i, M_grid, zob_j, mor)   # (NM,)
    wt = n_M * P * M_grid
    lnM = np.log(M_grid)
    num = np.trapz(wt * b_M, lnM)
    den = np.trapz(wt, lnM)
    return float(num / den) if den > 0.0 else float("nan")


def _ltr_weight(ltr_nodes, lob_i, zob_j, mor, M_grid, n_M,
                plob_params=None):
    """Weight prior for the ltr marginalisation.

    If ``plob_params`` is supplied (dict with splined EMG coefficients
    a_tau/b_tau/a_mu/b_mu/a_sig/b_sig/a_fprj/b_fprj as 1-D arrays over z),
    returns
        P(lob | ltr, zob) * int dM n(M, zob) P(ltr | M, zob) M
    -- equivalent to richness_selection.sel_bias.b_sel_marginalised with
    use_plob_ltr=True.

    If ``plob_params`` is None, returns the HOD prior alone (which
    falls back to a broad weight when P(ltr|M,z) is near-zero); this
    matches the legacy behaviour where ``use_plob_ltr=False``.
    """
    lnM = np.log(M_grid)
    # HOD prior integrated over mass: int dM n(M,z) P(ltr|M,z) M
    P = _p_ltr_given_M(ltr_nodes[:, None], M_grid[None, :], zob_j, mor)
    prior = np.trapz(n_M[None, :] * P * M_grid[None, :], lnM, axis=1)
    if plob_params is None:
        return prior

    # EMG in (lob | ltr, zob).  Parameters depend on zob and ltr.
    z_grid = plob_params["z"]
    # Interpolate each coefficient to zob
    def _eval(key):
        return float(np.interp(zob_j, z_grid, plob_params[key]))
    atau, btau = _eval("a_tau"), _eval("b_tau")
    amu,  bmu  = _eval("a_mu"),  _eval("b_mu")
    asig, bsig = _eval("a_sig"), _eval("b_sig")
    afprj, bfprj = _eval("a_fprj"), _eval("b_fprj")
    # closed forms from richness_selection.plob_ltr
    mu_arr    = amu + bmu * ltr_nodes
    sigma_arr = bsig * np.power(ltr_nodes, asig)
    tau_arr   = btau / np.power(ltr_nodes, atau)
    fprj_arr  = bfprj / np.power(1.0 + np.exp(-ltr_nodes), afprj)
    plob = _p_lob_given_ltr_emg(lob_i, ltr_nodes, zob_j,
                                 mu_arr, sigma_arr, tau_arr, fprj_arr)
    return prior * plob


def execute(block, config):
    t_start = time.perf_counter()
    theta       = config["theta"]
    zob         = config["zob"]
    lob         = config["lob"]
    ltr_lo        = config["ltr_lo"]
    ltr_hi_factor = config["ltr_hi_factor"]
    ltr_hi_fixed  = config["ltr_hi_fixed"]
    n_ltr         = config["n_ltr"]
    verbose       = config.get("verbose", False)

    N_zob = zob.size
    N_lam = lob.size

    # Per-lob ltr integration range matches Python
    # SelBias.b_sel_marginalised: [ltr_lo, ltr_hi_factor * lob] (default
    # factor = 3).  A fixed upper edge can be forced via [bsel] ltr_hi
    # > 0 for legacy behaviour.
    leg_nodes, leg_weights = np.polynomial.legendre.leggauss(n_ltr)
    def _ltr_grid_for_lob(lob_i):
        hi = ltr_hi_fixed if ltr_hi_fixed > 0.0 else ltr_hi_factor * lob_i
        nodes = 0.5 * (hi + ltr_lo) + 0.5 * (hi - ltr_lo) * leg_nodes
        wts   = leg_weights * 0.5 * (hi - ltr_lo)
        return nodes, wts

    # --- read the three CPP outputs ----------------------------------------
    # Layout in the ini: grid zipped as (zo, lambda_bin) with lambda_bin
    # inner; N_grid = N_zob * N_lam, flat index g = i_zob * N_lam + i_lam.
    # C++ BSelMargIntegrand now emits (P1, I1, J) with J = I2 - I1
    # computed directly to dodge catastrophic cancellation at large
    # theta (sigma -> 1).  Downstream we use J as the b_zero
    # denominator and reconstruct I2 = I1 + J only for diagnostics.
    P1_flat = _read_vec_1d(block, "b_sel_marg_P1", "vals").ravel()
    I1_flat = _read_vec_1d(block, "b_sel_marg_I1", "vals").ravel()
    J_flat  = _read_vec_1d(block, "b_sel_marg_J",  "vals").ravel()
    N_grid  = N_zob * N_lam
    assert P1_flat.size == N_grid, (P1_flat.shape, N_grid)

    P1 = P1_flat.reshape(N_zob, N_lam)
    I1 = I1_flat.reshape(N_zob, N_lam)
    J  = J_flat.reshape(N_zob, N_lam)
    I2 = I1 + J

    # --- chi(z) (for theta_lob) -- simple interpolant from distances ------
    # CosmoSIS distances/d_c is the comoving distance in Mpc (no h factor),
    # but the C++ p_operator / sigma_prj consumers compute their theta_lob
    # with chi in cMpc/h (see src/models/p_operator_t.hh line 256:
    # `chi_o = (*chi_).clamp(zob) * h0_`).  Multiply by h0 here so the
    # sigmoid we tabulate against `theta` is evaluated at the same theta_lob
    # the consumers use.
    # Bug: prior to 2026-05-19 this used raw d_c (Mpc), shifting the C++
    # b_sel_marginalised sigmoid by a factor of h0 in theta and producing
    # a 12% residual vs the Python richness_selection.sel_bias reference
    # at the wall-grid probe (lob=25, zob=0.275); see physics_validation_y1rerun.Rmd §2.7.
    h0 = float(block["cosmological_parameters", "h0"])
    z_of = _read_vec_1d(block, "distances", "z")
    chi_of = _read_vec_1d(block, "distances", "d_c") * h0   # cMpc/h
    def chi(zx): return np.interp(zx, z_of, chi_of)

    # --- MOR scalars (for the marginalisation weight) ----------------------
    # Accepts either the native (log10_Mmin, log10_M1) form or the
    # MCMC-friendly (log10_Mmin, log10_ratio) form, where
    # log10_ratio = log10(M1/Mmin).  Matches MOR_HOD_t in
    # src/models/mor_hod_t.hh and sel_function._read_mor in
    # src/modules/sel_function/sel_function.py:494.  log10_ratio wins
    # if both are present.
    log10_Mmin = float(block["cluster_mor", "log10_Mmin"])
    try:
        log10_ratio = float(block["cluster_mor", "log10_ratio"])
        log10_M1 = log10_Mmin + log10_ratio
    except Exception:
        log10_M1 = float(block["cluster_mor", "log10_M1"])
    mor = dict(
        log10_Mmin   = log10_Mmin,
        log10_M1     = log10_M1,
        alpha        = float(block["cluster_mor", "alpha"]),
        epsilon      = float(block["cluster_mor", "epsilon"]),
        sigma_lambda = float(block["cluster_mor", "sigma_lambda"]),
    )

    # --- HMF n(M, z) and bias b(M, z) from the datablock -----------------
    # mass_function publishes m_h and dndlnmh; halomodel publishes bias on
    # its own m_h/z grids.  Interpolate both onto a shared 100-point log
    # mass grid over [min_mass4integral, 10^ln_M_max_log10] matching
    # Python sel_bias.py:106-107.
    MIN_MASS = config["min_mass4integral"]    # 1e13 Msun/h
    MAX_LOG10M = config["ln_M_max_log10"]     # 15.5
    NM_BEFF = config.get("n_m_beff", 100)
    M_grid = 10.0 ** np.linspace(np.log10(MIN_MASS), MAX_LOG10M, NM_BEFF)

    mf_m = _read_vec_1d(block, "mass_function", "m_h")
    mf_z = _read_vec_1d(block, "mass_function", "z")
    mf_vals = np.asarray(block["mass_function", "dndlnmh"], dtype=float)
    # dndlnmh: shape (Nz, NM) per cosmosis convention
    if mf_vals.ndim == 1:
        mf_vals = mf_vals.reshape(mf_z.size, mf_m.size)
    # The CosmoSIS mf_tinker Fortran module stores the mass axis as
    # ``Mh = (4pi/3) * 2.775e11 * Rh^3`` (see compute_mf_tinker.f90,
    # module.yaml: "Mass in (omega_matter h^-1 M_solar)").  With the
    # constant 2.775e11 = rho_crit,0 * h^2 * Omega_m^-1 (i.e. it
    # omits the Omega_m factor), the stored Mh equals M_phys/Omega_m.
    # Rescale the axis back to physical M/h so downstream lookups at
    # M = 1e14 Msun/h hit the right bin.
    omm = float(block["cosmological_parameters", "omega_m"])
    try:
        omn = float(block["cosmological_parameters", "omega_nu"])
    except Exception:
        omn = 0.0
    mf_m = mf_m * (omm - omn)

    hm_m = _read_vec_1d(block, "halomodel", "m_h")
    hm_z = _read_vec_1d(block, "halomodel", "z")
    hm_bias = np.asarray(block["halomodel", "bias"], dtype=float)
    if hm_bias.ndim == 1:
        hm_bias = hm_bias.reshape(hm_z.size, hm_m.size)

    def _interp_Mz(vals, m_axis, z_axis, M_target, z_target):
        # bilinear in (log M, z)
        logM_t = np.log(M_target)
        logm_a = np.log(m_axis)
        out = np.empty(M_target.size)
        iz = np.searchsorted(z_axis, z_target)
        iz = int(np.clip(iz, 1, z_axis.size - 1))
        fz = (z_target - z_axis[iz - 1]) / (z_axis[iz] - z_axis[iz - 1] + 1e-30)
        for k, lm in enumerate(logM_t):
            jm = int(np.clip(np.searchsorted(logm_a, lm), 1, logm_a.size - 1))
            fm = (lm - logm_a[jm - 1]) / (logm_a[jm] - logm_a[jm - 1] + 1e-30)
            v = ((1 - fz) * ((1 - fm) * vals[iz - 1, jm - 1]
                             + fm * vals[iz - 1, jm])
                 + fz * ((1 - fm) * vals[iz, jm - 1]
                         + fm * vals[iz, jm]))
            out[k] = v
        return out

    # plob_ltr EMG parameters pre-loaded from npz (see setup()).  May be
    # None, in which case the ltr marginalisation uses a flat prior
    # (use_plob_ltr=False).
    plob_params = config.get("plob_params")

    # --- KEY FACTORISATION ---------------------------------------------
    # b_sel(theta | lob, ltr, zob) = b_zero(ltr) + [b_infty(ltr) - b_zero(ltr)]
    #                                              * sigmoid(theta | lob, zob)
    # Only b_zero and b_infty depend on ltr; only sigmoid depends on theta.
    # The ltr-marginalisation weight w(ltr | lob, zob) is theta-independent.
    # So the marginalised answer separates analytically:
    #     <b_sel>(theta) = B_small + [B_large - B_small] * sigmoid(theta)
    # with two scalars per (lob, zob):
    #     B_small = <b_zero>_{ltr}     (small-scale / theta -> 0 asymptote)
    #     B_large = <b_infty>_{ltr}    (large-scale / theta -> infty asymptote).
    # "infty" is retired in the public API in favour of "large" since
    # physically the saturated regime is the large-scale limit.
    # Downstream C++ reads these two scalars + the sigmoid formula and
    # evaluates b_sel(theta) analytically, sidestepping the spline
    # artefacts of the old 32-node theta tabulation.
    # ----------------------------------------------------------------------
    N_theta = theta.size
    vals = np.zeros((N_lam, N_zob, N_theta))  # kept for backward compat
    b_eff_grid   = np.zeros((N_zob, N_lam))
    B_small_grid = np.zeros((N_zob, N_lam))   # <b_zero>_ltr   (small-scale)
    B_large_grid = np.zeros((N_zob, N_lam))   # <b_infty>_ltr  (large-scale)

    for j, zo in enumerate(zob):
        # `mass_function/dndlnmh` is dn/dlnM in (Mpc/h)^-3.  Downstream
        # `_compute_b_eff` and `_ltr_weight` expect `n_M = dn/dM`
        # (matching the Python `richness_selection.HMF.__call__`
        # convention), which they multiply by `M_grid` inside the
        # integrand.  So divide by M here to drop the extra factor of
        # M that would otherwise slip in.
        n_Mj = _interp_Mz(mf_vals, mf_m, mf_z, M_grid, zo) / M_grid
        b_Mj = _interp_Mz(hm_bias, hm_m, hm_z, M_grid, zo)
        th_lob = _theta_lob(lob[:, None], zo, chi)    # (N_lam,1)
        sigma_t = _sigmoid(theta[None, :], th_lob)    # (N_lam, N_theta)
        for i in range(N_lam):
            lob_i  = lob[i]
            # proper b_eff from mass-weighted integral
            b_eff_ji = _compute_b_eff(lob_i, zo, mor, M_grid, n_Mj, b_Mj)
            b_eff_grid[j, i] = b_eff_ji

            # Per-lob ltr GL grid on [ltr_lo, 3*lob_i] (matches Python
            # SelBias.b_sel_marginalised).  Rebuilt per bin because the
            # upper edge depends on lob_i.
            ltr_nodes, ltr_weights = _ltr_grid_for_lob(float(lob_i))

            Delta_RND = P1[j, i] + b_eff_ji * I2[j, i]
            denom     = J[j, i]    # = I2 - I1, computed directly in C++
            dpr       = (lob_i - ltr_nodes) / Delta_RND - 1.0
            b_infty_l = b_eff_ji * (1.0 + 0.13 * dpr)
            # b_zero closure (paper Eq.~b_LOS_latent).  If J = I2 - I1
            # collapses, the closed form is genuinely undefined --
            # surface NaN so the validator flags it instead of silently
            # rolling b_LOS into b_LSS (which would mask a real
            # cancellation in a future cosmology).
            b_zero_l  = np.where(
                np.abs(denom) > 1e-12 * (np.abs(I1[j, i]) + np.abs(I2[j, i])),
                ((lob_i - ltr_nodes) - P1[j, i]
                 - b_infty_l * I1[j, i]) / denom,
                np.nan,
            )
            # Marginalisation weight: HOD prior optionally folded with
            # the EMG P(lob | ltr, zob).
            w_prior = _ltr_weight(ltr_nodes, lob_i, zo, mor,
                                  M_grid, n_Mj, plob_params)
            w = ltr_weights * w_prior
            den = w.sum()
            if den > 0.0:
                B_small = float((w * b_zero_l).sum() / den)
                B_large = float((w * b_infty_l).sum() / den)
            else:
                B_small = 0.0
                B_large = 0.0
            B_small_grid[j, i] = B_small
            B_large_grid[j, i] = B_large

            # Tabulated form (kept for backward compat; downstream C++
            # should use B_small + (B_large - B_small) * sigmoid directly).
            vals[i, j] = B_small + (B_large - B_small) * sigma_t[i]

    block["b_sel_marginalised", "lob"]   = lob
    block["b_sel_marginalised", "zob"]   = zob
    block["b_sel_marginalised", "theta"] = theta
    block["b_sel_marginalised", "vals"]  = vals
    # Expose b_eff(zob, lob) for diagnostics / downstream consumers.
    block["b_sel_marginalised", "b_eff"] = b_eff_grid
    # Two-scalar factorisation used by RedShearPrj_evaluator.  Shape
    # matches b_eff (N_zob, N_lam).  Consumers should compute
    #     b_sel(theta) = B_small + (B_large - B_small) * sigmoid(theta)
    # with the Costanzi-2026 Eq.~6 sigmoid evaluated at the true
    # (lob, zob) -- no theta tabulation, no nearest-zob tie-break.
    block["b_sel_marginalised", "b_small"] = B_small_grid
    block["b_sel_marginalised", "b_large"] = B_large_grid

    if verbose:
        status = ("use_plob_ltr=True" if plob_params is not None
                  else "use_plob_ltr=False")
        print(f"[bsel.py] b_sel_marginalised ({N_lam}x{N_zob}x{N_theta}, "
              f"{status}): {(time.perf_counter() - t_start)*1000:.1f} ms",
              flush=True)

        # Report the ltr-marginalised small-/large-scale bias asymptotes,
        # plus b_eff.  These are the physically meaningful summary
        # numbers per (lob, zob) bin; the full b_sel(theta) is recovered
        # downstream by the Costanzi-2026 sigmoid
        #   b_sel(theta) = B_small + (B_large - B_small) * sigmoid(theta).
        print("[bsel.py] marginalised bias asymptotes (rows: zob, cols: lob)")
        lob_header = "  zob \\ lob " + "  ".join(f"{x:>8.1f}" for x in lob)
        def _show(name, grid):
            print(f"  {name}:")
            print(f"    {lob_header}")
            for j, zo in enumerate(zob):
                row = "  ".join(f"{grid[j, i]:>+8.3f}" for i in range(N_lam))
                print(f"    zob={zo:.3f}  {row}")
        _show("b_eff   (= b_halo at M[lob])",     b_eff_grid)
        _show("B_small (small-scale asymptote)",  B_small_grid)
        _show("B_large (large-scale asymptote)",  B_large_grid)
    return 0


def cleanup(config):
    return 0
