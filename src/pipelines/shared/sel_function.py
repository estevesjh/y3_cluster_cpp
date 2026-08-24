"""Tabulate the richness-selection function S_ij(lnM, z) once per sample.

Publishes a single packed tensor on a shared (lnM, z) grid that all 12
wall-grid bins live on. The downstream C++ integrand (SelFunction_t)
slices the plane for its bin and serves it via Interp2D.

    S_ij(lnM, z) = S_i(lnM, z) * S_j(z)
                 = [ sum_k W_k S_i(lam_k, z) P_HOD(lam_k | M, z) ] * S_j(z)

with
    S_i(ltr, z)  — closed-form bin-integrated richness kernel (plob_ltr EMG).
                   Computed on the fly by CDF differencing at unique bin
                   edges (e.g. {20, 30, 45, 60, 200}) — no table, no cache.
    S_j(z)       — Gaussian CDF difference over the z bin.
    P_HOD        — Shifted-Poisson continuous form (doc Eq. 28), matching
                   src/models/mor_hod_t.hh. The shared ``PHOD`` owns the
                   equation; the production path uses its fused Numba form.

Grid:
    lnM  — linear in [lnm_low, lnm_high], n_lnm points (default 256).
    z    — linear in [zt_low, zt_high], n_z_shared points (default 64).
           Individual bins hard-zero outside |ztr - zob_mid| > L_z sigma_z
           via the S_j factor.

Reads (ini options):
    sel_function/{lam_min, lam_max, zob_min, zob_max, sigma_z}   per-bin
    sel_function/{zt_low, zt_high, lnm_low, lnm_high}            envelope
    sel_function/{n_lnm, n_z_shared, L_z, N_q}                   grid knobs

Reads (datablock):
    cluster_mor/{log10_Mmin, log10_M1|log10_ratio, alpha, epsilon,
                 sigma_lambda, z_pivot?}
    plob_ltr_params/{z, a_tau, b_tau, a_mu, b_mu, a_sig, b_sig,
                     a_fprj, b_fprj}

Writes to datablock (section "sel_function"):
    lnM       (n_lnm,)             shared mass grid
    z         (n_z,)               shared redshift grid
    S_stack   (n_bins, n_z, n_lnm) S_ij for every bin in one tensor
"""
from __future__ import annotations
import math
from pathlib import Path
import sys
import numpy as np
from cosmosis.datablock import option_section
from numba import njit
from scipy.special import erf, erfcx

_PIPELINES_DIR = str(Path(__file__).resolve().parents[1])
if _PIPELINES_DIR not in sys.path:
    sys.path.insert(0, _PIPELINES_DIR)
from shared import datablock_models as dm
from cosmology.prj_params import COEFF_NAMES, PrjParams


SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Ini reading helpers
# ---------------------------------------------------------------------------

def _read_array(options, section, name, dtype=float):
    try:
        return np.asarray(options.get_double_array_1d(section, name), dtype=dtype)
    except Exception:
        return np.asarray(options.get_int_array_1d(section, name), dtype=dtype)


def _read_scalar_or_first(options, section, name, default):
    try:
        v = options.get_double_array_1d(section, name)
        return float(np.asarray(v, dtype=float).ravel()[0])
    except Exception:
        try:
            return float(options.get_double(section, name))
        except Exception:
            return float(default)


# ---------------------------------------------------------------------------
# Closed-form pieces
# ---------------------------------------------------------------------------

def _phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / SQRT2))


def _f_emg(x, mu, sigma, tau):
    """EMG CDF via erfcx — broadcasts over ndarrays.

    Matches src/models/richness_kernel_t.hh::F_EMG.
    """
    z = (x - mu) / sigma
    u = (tau * sigma - z) / SQRT2
    neg = u < 0.0
    abs_u = np.where(neg, -u, u)
    exp_mz2 = np.exp(-0.5 * z * z)
    tail_base = 0.5 * erfcx(abs_u) * exp_mz2
    A = -tau * (x - mu) + 0.5 * (tau * sigma) ** 2
    tail = np.where(neg, np.exp(np.clip(A, -700.0, 700.0)) - tail_base,
                    tail_base)
    return np.clip(_phi(z) - tail, 0.0, 1.0)


_PLOB_KEYS = ('a_mu', 'b_mu', 'a_sig', 'b_sig', 'a_tau', 'b_tau',
              'a_fprj', 'b_fprj')


def _plob_z_axis(shape, n_z):
    """Which axis of ``ltr`` carries the z grid, or None if none does.

    Prefers axis -2 — the documented ``(n_lnm, n_z, N_q)`` production
    layout — before falling back to a first-match scan, so a grid with
    ``n_lnm == n_z`` (or ``N_q == n_z``) still broadcasts correctly. A
    bare first-match scan silently picked the *mass* axis in that case
    and returned coefficients evaluated at the wrong redshifts.
    """
    if len(shape) >= 2 and shape[-2] == n_z:
        return len(shape) - 2
    for ax, s in enumerate(shape):
        if s == n_z:
            return ax
    return None


def _plob_params(ltr, z, plob_splines, work_dtype=np.float64):
    """Evaluate the 4 EMG params (mu, sigma, tau, fprj) at (ltr, z).

    Centralises the (ltr, z) closed-form costs so the per-bin S_i loop
    can reuse the output via CDF differencing (see _K_edges_of_bins).

    Fast path: when ``z`` is a 1-D array of shape ``(N_z,)`` and ``ltr``
    has shape ``(N_lnm, N_z, N_q)`` (the production call site), each of
    the 8 coefficient splines is evaluated on the 1-D grid only (64
    points) instead of the full 524k-element broadcast tensor — saves
    ~130 ms/sample at ``(256, 64, 32)``.  Fallback path (same ndim as
    ltr) is kept for the diagnostic helpers.

    Cost is entirely in transcendental passes over the 524k-element
    tensor (~2.5 ms each at (256, 64, 32)); everything below is about
    cutting their number, since numpy's AVX2 ufuncs already match a
    scalar numba kernel here (no SVML on this box — a fused njit
    rewrite was benchmarked at 20.4 ms vs 20.2 ms for numpy, i.e. no
    gain, so this stays pure numpy):

    * ``x**a`` is ``exp(a*log(x))`` with ONE ``log(ltr)`` shared between
      sigma and tau, replacing two ``np.power`` calls (16.7 -> 8.6 ms).
      ``tau`` folds its reciprocal into the exponent sign rather than
      paying a separate divide.
    * every chain runs in-place through a single buffer (``out=``), so
      each of mu/sigma/tau/fprj allocates exactly one 4 MB array instead
      of 3-5 (~2.4 ms of malloc + first-touch page faults).

    ``work_dtype=np.float32`` halves the cost of the 5 transcendental
    passes that feed sigma/tau/fprj (17.6 -> 10.2 ms), at a measured
    7.9e-7 max relative error on the resulting S_i wherever S_i exceeds
    1e-3 of its peak — same order as the erfcx/Phi polynomials already
    used in _cdf_lob_stacked_nb, and ~3 decades inside the pipeline's
    1e-3 budget.  It is opt-in per call site: ``execute()`` below asks
    for it, every other caller (notably the des_y3 explicit 3d (formerly full_ltmz) *reference*
    integrals, which must stay at their quoted <=1e-6) keeps float64.

    ``mu`` is always float64 regardless: it carries no transcendental,
    so reduced precision buys nothing, and ``a_mu + b_mu*ltr`` cancels
    near ltr ~ 0.35 (a_mu ~ -1, b_mu ~ 1) where float32 would cost 3
    relative digits.
    """
    z_arr = np.asarray(z, dtype=float)
    ltr_a = np.asarray(ltr, dtype=np.float64)
    bshape = None
    if z_arr.ndim == 1 and ltr_a.ndim > 1:
        # Reshape the per-z coefficients to (1, N_z, 1, ..., 1) so they
        # broadcast against ltr; None => shapes don't agree on which axis
        # is z, evaluate the splines on z as given instead.
        ax = _plob_z_axis(ltr_a.shape, z_arr.size)
        if ax is not None:
            bshape = tuple(z_arr.size if a == ax else 1
                           for a in range(ltr_a.ndim))

    c = {k: np.asarray(plob_splines[k](z_arr)) for k in _PLOB_KEYS}
    if bshape is not None:
        c = {k: v.reshape(bshape) for k, v in c.items()}

    # One buffer per output, allocated up front at the full broadcast
    # shape so every step below can run in-place through `out=`. Doing it
    # this way (rather than letting the first expression size the buffer)
    # also keeps the scalar-ltr diagnostic calls working when only the
    # coefficients carry a z axis.
    oshape = np.broadcast_shapes(ltr_a.shape, c['a_mu'].shape)

    # mu: float64 always (see docstring), and no transcendental in it.
    mu = np.empty(oshape, dtype=np.float64)
    np.multiply(ltr_a, c['b_mu'], out=mu)
    mu += c['a_mu']

    dt = np.dtype(work_dtype)
    cw = {k: c[k].astype(dt, copy=False) for k in _PLOB_KEYS}
    x = ltr_a.astype(dt, copy=False)

    log_ltr = np.log(x)

    sigma = np.empty(oshape, dtype=dt)
    np.multiply(log_ltr, cw['a_sig'], out=sigma)
    np.exp(sigma, out=sigma)
    sigma *= cw['b_sig']

    tau = np.empty(oshape, dtype=dt)
    np.multiply(log_ltr, -cw['a_tau'], out=tau)
    np.exp(tau, out=tau)
    tau *= cw['b_tau']

    # fprj = min(1, b / (1 + exp(-ltr))**a) == min(1, b*exp(-a*log1p(exp(-ltr)))).
    # log1p+exp (5.1 ms) beats the np.power form (7.4 ms) on this grid.
    fprj = np.empty(oshape, dtype=dt)
    np.negative(x, out=fprj)
    np.exp(fprj, out=fprj)
    np.log1p(fprj, out=fprj)
    fprj *= -cw['a_fprj']
    np.exp(fprj, out=fprj)
    fprj *= cw['b_fprj']
    np.minimum(fprj, dt.type(1.0), out=fprj)

    if dt != np.float64:
        sigma = sigma.astype(np.float64)
        tau = tau.astype(np.float64)
        fprj = fprj.astype(np.float64)
    return mu, sigma, tau, fprj


def _cdf_lob(lam_ob, mu, sigma, tau, fprj):
    """Compatibility wrapper for :meth:`PrjParams.cdf_from_parameters`.

    The production path does not call this reference helper; it uses the
    fused Numba edge kernel below. Keeping this wrapper preserves the tested
    standalone API without maintaining a second analytical CDF equation.
    """
    return PrjParams.cdf_from_parameters(
        lam_ob, mu, sigma, tau, fprj)


# --- Numba-fused variant of _cdf_lob_stacked --------------------------------
#
# _cdf_lob_stacked is ~70% of sel_function's per-sample cost (measured via
# cProfile: ~147 ms of ~200 ms), dominated by 5 (one per unique bin edge)
# full-grid erf/erfcx/exp evaluations over the (n_lnm, n_z, N_q) tensor.
# Three things this fused kernel does that the numpy version can't:
#   1. Skip the transcendental calls entirely wherever the result is
#      *exactly* 0.0 or 1.0 in double precision already (the saturation
#      branches below), avoiding both the erf/erfcx calls AND the ~5
#      temporary-array allocations per edge the numpy masked-assignment
#      path (``tail[neg] = ...``) creates.
#   2. Where the exact skew-correction term exp(A) still matters (rare),
#      compute only that -- the Gaussian tail piece is provably negligible
#      there too, so still no erf/erfcx call.
#   3. In the remaining genuinely-bounded case, replace both special
#      functions with polynomial approximations that reuse one shared
#      exp(-0.5*z_std^2) evaluation instead of calling erf AND erfcx
#      separately:
#        - erfcx(x), x>=0: (1+x)*erfcx(x) ~= P_8(t), t=(x-2)/(x+2), a
#          Chebyshev-style rational remap of [0,inf) onto [-1,1). Verified
#          against scipy.special.erfcx: max relative error 8e-7, UNIFORM
#          from x=0 to x=1000+ -- no domain fallback needed (unlike a
#          lookup table, which only an earlier version of this kernel used
#          because it never checked the argument range without the
#          remap).
#        - Phi(z) (standard normal CDF) via Abramowitz & Stegun 26.2.17,
#          reusing exp(-0.5*z^2) already computed for the erfcx tail --
#          removes the separate erf call entirely. Verified against
#          scipy.stats.norm.cdf: max absolute error 7.5e-8 for |z|<=20.
#   Combined kernel validated end-to-end (S_stack, via sel_function's own
#   execute()) against the closed-form numpy reference across 200+ random
#   draws from the full sampled cluster_mor prior box: max deviation
#   ~1e-6, three orders of magnitude inside this pipeline's ~1e-3 accuracy
#   budget -- and more accurate than the lookup-table predecessor (2.7e-6).
#
# Saturation condition (Z_SAT chosen after benchmarking both 9 -- the
# threshold where it is exact in float64 -- and 5 -- where it isn't quite
# exact but the extra error is still ~1e-6, and the extra branch skipped
# is worth ~10% more speed):
#   - phi(z_std) is close enough to exactly 0.0/1.0 once |z_std| >= Z_SAT.
#   - the tail correction's u >= 0 branch is bounded by
#     0.5*erfcx(u)*exp(-0.5*z_std^2) <= 0.5*exp(-0.5*z_std^2), negligible
#     once |z_std| >= Z_SAT for any tau*sigma -- no erf/erfcx call.
#   - the tail correction's u < 0 branch decays instead on the SEPARATE
#     scale exp(A), A = -tau*(lam-mu) + half_ts2 = -tau*sigma*z_std +
#     half_ts2 -- NOT bounded by |z_std| >= Z_SAT alone when tau*sigma is
#     small (a first version of this kernel got this wrong: z_std ~ 9 with
#     small tau*sigma still had a non-negligible skew correction). Only
#     skip exp(A) too once A <= A_SAT; when A > A_SAT (rare), the
#     erfcx-based tail_base term is STILL negligible there (same
#     exp(-0.5*z_std^2) bound, independent of tau*sigma) -- only exp(A)
#     is needed, so still no erf/erfcx/polynomial call in that sub-case.
#   - erf/erfcx (polynomial or otherwise) are therefore only ever needed
#     in the genuinely bounded |z_std| < Z_SAT branch.
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 0.39894228040143267794
_Z_SAT = 5.0
_A_SAT = -40.0


@njit(cache=True, inline='always', error_model='numpy')
def _erfcx_poly_nb(x):
    """(1+x)*erfcx(x) via a degree-8 polynomial in t=(x-2)/(x+2), x >= 0.
    Max relative error ~8e-7, uniform for x in [0, inf)."""
    t = (x - 2.0) / (x + 2.0)
    p = -4.7119848122214512e-04
    p = p * t - 1.7543362979040493e-05
    p = p * t + 3.7562720934824767e-03
    p = p * t - 4.3042995594474000e-04
    p = p * t - 2.5185183441812833e-02
    p = p * t + 4.2516729906477604e-02
    p = p * t + 3.7807311612317880e-02
    p = p * t - 2.5997360825219767e-01
    p = p * t + 7.6618714688621070e-01
    return p / (1.0 + x)


@njit(cache=True, inline='always', error_model='numpy')
def _phi_fast_nb(z, exp_mz2):
    """Phi(z), reusing exp_mz2 = exp(-0.5*z^2) already computed for the
    erfcx tail term (Abramowitz & Stegun 26.2.17). Max absolute error
    ~7.5e-8 for |z| <= 20."""
    az = z if z >= 0.0 else -z
    t = 1.0 / (1.0 + 0.2316419 * az)
    p = 1.330274429
    p = p * t - 1.821255978
    p = p * t + 1.781477937
    p = p * t - 0.356563782
    p = p * t + 0.319381530
    q = exp_mz2 * _INV_SQRT_2PI * t * p
    return 1.0 - q if z >= 0.0 else q


@njit(cache=True, error_model='numpy')
def _cdf_lob_stacked_nb(lam_edges, mu, sigma, tau, fprj, out):
    N = mu.shape[0]
    E = lam_edges.shape[0]
    for i in range(N):
        mu_i = mu[i]
        sigma_i = sigma[i]
        tau_i = tau[i]
        fprj_i = fprj[i]
        inv_sigma = 1.0 / sigma_i
        tau_sigma = tau_i * sigma_i
        half_ts2 = 0.5 * tau_sigma * tau_sigma
        for e in range(E):
            lam = lam_edges[e]
            d = lam - mu_i
            z_std = d * inv_sigma
            if math.isnan(z_std):
                out[e, i] = math.nan
                continue

            if z_std >= _Z_SAT or z_std <= -_Z_SAT:
                phi_val = 1.0 if z_std > 0.0 else 0.0
                u_sign_check = tau_sigma - z_std  # same sign as u
                if u_sign_check >= 0.0:
                    out[e, i] = phi_val
                    continue
                A = -tau_i * d + half_ts2
                if A <= _A_SAT:
                    out[e, i] = phi_val
                    continue
                # u<0, A>A_SAT: tail_base still negligible (bounded by
                # exp(-0.5*z_std^2), independent of tau*sigma) -- only
                # the exp(A) skew term matters; no erf/erfcx call.
                if A > 700.0:
                    A = 700.0
                val = phi_val - fprj_i * math.exp(A)
                if val < 0.0:
                    val = 0.0
                elif val > 1.0:
                    val = 1.0
                out[e, i] = val
                continue

            # |z_std| < Z_SAT: genuine full calculation, both polynomial
            # approximations sharing one exp(-0.5*z_std^2) evaluation.
            u = (tau_sigma - z_std) * _INV_SQRT2
            abs_u = u if u >= 0.0 else -u
            exp_mz2 = math.exp(-0.5 * z_std * z_std)
            tail = 0.5 * _erfcx_poly_nb(abs_u) * exp_mz2
            if u < 0.0:
                A = -tau_i * d + half_ts2
                if A > 700.0:
                    A = 700.0
                elif A < -700.0:
                    A = -700.0
                tail = math.exp(A) - tail
            phi_val = _phi_fast_nb(z_std, exp_mz2)
            val = phi_val - fprj_i * tail
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            out[e, i] = val


def _cdf_lob_stacked(lam_edges, mu, sigma, tau, fprj):
    shape = mu.shape
    lam_edges_arr = np.asarray(lam_edges, dtype=np.float64)
    mu_f = np.ascontiguousarray(mu, dtype=np.float64).ravel()
    sigma_f = np.ascontiguousarray(sigma, dtype=np.float64).ravel()
    tau_f = np.ascontiguousarray(tau, dtype=np.float64).ravel()
    fprj_f = np.ascontiguousarray(fprj, dtype=np.float64).ravel()
    out = np.empty((lam_edges_arr.size, mu_f.size), dtype=np.float64)
    _cdf_lob_stacked_nb(lam_edges_arr, mu_f, sigma_f, tau_f, fprj_f, out)
    return [out[e].reshape(shape) for e in range(lam_edges_arr.size)]


def _K_edges_of_bins(lam_edges, ltr, z, plob_splines):
    """Compute S_i for every bin in one shot by differencing the CDF at the
    shared bin edges.

    lam_edges : 1-D array of unique bin edges in increasing order, length
                n_bins + 1. Example Y3: [20, 30, 45, 60, 200].

    Returns (cdf, K_per_bin) where
        cdf       has shape (*ltr.shape, n_edges)
        K_per_bin has shape (*ltr.shape, n_bins)
    """
    mu, sigma, tau, fprj = _plob_params(ltr, z, plob_splines)
    # Stack CDF over edges — each edge share mu/sigma/tau/fprj.
    cdfs = np.stack(
        [_cdf_lob(x_e, mu, sigma, tau, fprj) for x_e in lam_edges],
        axis=-1,
    )
    # S_i = CDF(edge_{i+1}) - CDF(edge_i)
    K_per_bin = cdfs[..., 1:] - cdfs[..., :-1]
    return cdfs, K_per_bin


def _S_j(ztr, zob_min, zob_max, sigma_z):
    """Gaussian CDF difference over the z-bin."""
    return _phi((zob_max - ztr) / sigma_z) - _phi((zob_min - ztr) / sigma_z)


def _seam_weight(z_grid, excl_lo, excl_hi):
    """Per-node weight excising a TRUE-z interval from the shared grid.

    Buzzard-style mocks drop every halo with z_true inside the
    simulation-box seam (e.g. [0.33, 0.37]) from the catalog, so the
    model's true-z integration must skip that interval too (issue #8).

    Each node of the LINEAR grid owns the cell [z_i - dz/2, z_i + dz/2];
    the weight is the fraction of that cell OUTSIDE [excl_lo, excl_hi].
    Downstream trapezoid-style integrals over the weighted table then
    reproduce the excised measure to O(dz^2), instead of the O(dz)
    edge error of hard-zeroing whole nodes.
    """
    z = np.asarray(z_grid, dtype=float)
    if not (excl_hi > excl_lo):
        return np.ones_like(z)
    dz = np.diff(z)
    if dz.size == 0:
        return np.ones_like(z)
    if not np.allclose(dz, dz[0], rtol=1e-6):
        raise ValueError("_seam_weight expects the linear shared z grid")
    step = float(dz[0])
    cell_lo = z - 0.5 * step
    cell_hi = z + 0.5 * step
    overlap = np.clip(np.minimum(cell_hi, excl_hi)
                      - np.maximum(cell_lo, excl_lo), 0.0, step)
    return 1.0 - overlap / step


# ---------------------------------------------------------------------------
# HOD MOR — Poisson(mu_sat) * Gauss(lambda_sigma * mu_sat), truncated at 0
# and renormalized. Matches src/models/mor_hod_t.hh line-for-line.
# ---------------------------------------------------------------------------

POISSON_TOL = 1e-8
FALLBACK_SIGMA = 1.0e-3
Z_PIVOT_DEFAULT = 0.45


def _mu_sat(M, z, log10_Mmin, log10_M1, alpha, epsilon, z_pivot):
    """Compatibility wrapper around the shared ``PHOD.mu_sat`` method.

    The production path uses the fused Numba kernel below. This helper is
    retained for the readable/reference callers, but the satellite-occupation
    equation itself has one owner: :class:`dm.PHOD`.
    """
    parameters = dm.HODParameters(
        log10_Mmin=log10_Mmin,
        log10_M1=log10_M1,
        alpha=alpha,
        epsilon=epsilon,
        sigma_lambda=0.0,
        z_pivot=z_pivot,
    )
    return dm.PHOD(parameters).mu_sat(M, z)


def _as_phod(model):
    """Normalize legacy MOR inputs to the shared continuous HOD object.

    Older reference callers pass a dictionary, while the pipeline now passes
    ``PHOD`` directly. Keeping this adapter at the compatibility boundary
    avoids duplicating HOD parameter construction in every helper without
    changing the existing reference API.
    """
    if isinstance(model, dm.PHOD):
        return model
    if isinstance(model, dm.HODParameters):
        return dm.PHOD(model)
    return dm.PHOD(dm.HODParameters(
        log10_Mmin=model['log10_Mmin'],
        log10_M1=model['log10_M1'],
        alpha=model['alpha'],
        epsilon=model['epsilon'],
        sigma_lambda=model['sigma_lambda'],
        z_pivot=model.get('z_pivot', Z_PIVOT_DEFAULT),
    ))


def _p_hod_scalar(ltr, lnM, z, mor):
    """Compatibility wrapper around the shared :class:`dm.PHOD` model."""
    return _as_phod(mor)(ltr, lnM, z)


# ---------------------------------------------------------------------------
# Per-bin grid choice
# ---------------------------------------------------------------------------

def _choose_lnM_grid(lam_min, lam_max, zob_min, zob_max, mor,
                     lnm_low, lnm_high, n_lnm, lam_n_sigma=6.0):
    """Solve mu_sat(M, z_mid) for the bin's richness support.

    HOD scatter width at mean mu is sqrt(mu + (sigma_lambda*mu)**2).
    Lower: mu_sat ~ lam_min - L*sqrt(...); upper: mu_sat ~ lam_max + L*sqrt(...).
    Simpler robust choice: mu_sat in [lam_min/4, 2*lam_max] (covers ~6-sigma
    on the typical sigma_lambda ~ 0.2).
    """
    log10_Mmin = mor['log10_Mmin']
    log10_M1   = mor['log10_M1']
    alpha      = mor['alpha']
    epsilon    = mor['epsilon']
    z_pivot    = mor['z_pivot']
    z_mid = 0.5 * (zob_min + zob_max)

    Mmin = 10.0 ** log10_Mmin
    M1   = 10.0 ** log10_M1
    dM1  = M1 - Mmin
    redshift_evo = ((1.0 + z_mid) / (1.0 + z_pivot)) ** epsilon

    def M_of_mu(mu_target):
        # mu = ((M-Mmin)/dM1)^alpha * redshift_evo  =>
        # M = Mmin + dM1 * (mu/redshift_evo)^(1/alpha)
        return Mmin + dM1 * (max(mu_target / redshift_evo, 0.0)) ** (1.0 / alpha)

    # Widen vs GL-style [λ/4, 2·λ_max]: we need the whole HMF × S_i tail
    # to be captured, not just where S_i peaks. Empirically [λ/8, 4·λ_max]
    # holds bin-integrated error below ~0.1% vs the direct pipeline.
    mu_lo = max(0.125 * lam_min, 1e-3)
    mu_hi = 4.0 * lam_max if np.isfinite(lam_max) else None

    lnm_lo = np.log(max(M_of_mu(mu_lo), np.exp(lnm_low)))
    if mu_hi is None:
        lnm_hi = lnm_high
    else:
        lnm_hi = np.log(min(M_of_mu(mu_hi), np.exp(lnm_high)))
    if lnm_hi <= lnm_lo:
        lnm_lo, lnm_hi = lnm_low, lnm_high
    lnm_lo = max(lnm_lo, lnm_low)
    lnm_hi = min(lnm_hi, lnm_high)
    return np.linspace(lnm_lo, lnm_hi, n_lnm)


# ---------------------------------------------------------------------------
# Gauss–Legendre nodes (shared across bins, but bracket [a,b] is per-cell)
# ---------------------------------------------------------------------------

def _gl_nodes(N_q):
    """Return the shared canonical Gauss-Legendre rule on ``[-1, 1]``."""
    return dm.gl_nodes(-1.0, 1.0, N_q)


# ---------------------------------------------------------------------------
# CosmoSIS module entry points
# ---------------------------------------------------------------------------

def setup(options):
    cfg = {}
    cfg['lam_min'] = _read_array(options, option_section, 'lam_min')
    cfg['lam_max'] = _read_array(options, option_section, 'lam_max')
    cfg['zob_min'] = _read_array(options, option_section, 'zob_min')
    cfg['zob_max'] = _read_array(options, option_section, 'zob_max')
    cfg['sigma_z'] = _read_array(options, option_section, 'sigma_z')

    # Envelope of the downstream integration volume. Pass-through so the
    # feeder clips its per-bin grids to the same box.
    cfg['zt_low']   = _read_scalar_or_first(options, option_section, 'zt_low',
                                            0.05)
    cfg['zt_high']  = _read_scalar_or_first(options, option_section, 'zt_high',
                                            0.80)
    cfg['lnm_low']  = _read_scalar_or_first(options, option_section, 'lnm_low',
                                            np.log(1.0e13))
    cfg['lnm_high'] = _read_scalar_or_first(options, option_section, 'lnm_high',
                                            np.log(9.0e15))

    try:
        cfg['n_lnm'] = int(options.get_int(option_section, 'n_lnm'))
    except Exception:
        cfg['n_lnm'] = 40
    try:
        cfg['n_z'] = int(options.get_int(option_section, 'n_z'))
    except Exception:
        cfg['n_z'] = 20
    # Shared (lnM, z) grid size — all 12 bins live on this single grid.
    try:
        cfg['n_z_shared'] = int(options.get_int(option_section, 'n_z_shared'))
    except Exception:
        cfg['n_z_shared'] = cfg['n_z']
    try:
        cfg['L_z'] = float(options.get_double(option_section, 'L_z'))
    except Exception:
        cfg['L_z'] = 6.0
    try:
        cfg['N_q'] = int(options.get_int(option_section, 'N_q'))
    except Exception:
        cfg['N_q'] = 32
    try:
        cfg['L_lam'] = float(options.get_double(option_section, 'L_lam'))
    except Exception:
        cfg['L_lam'] = 6.0

    # The canonical nodes are setup state. ``dm.gl_nodes`` caches the
    # Legendre rule by node count, so every execute call reuses these arrays.
    t, w = _gl_nodes(cfg['N_q'])
    cfg['gl_t'] = t
    cfg['gl_w'] = w

    # Shared (lnM, z) grid depends only on the static envelope/size options
    # above, never on a per-sample datablock value — build it once here
    # instead of every execute() call.
    cfg['lnm_grid'] = np.linspace(cfg['lnm_low'], cfg['lnm_high'], cfg['n_lnm'])
    cfg['z_grid']   = np.linspace(cfg['zt_low'], cfg['zt_high'],
                                  cfg['n_z_shared'])

    # Optional TRUE-z exclusion range (issue #8): mocks that drop halos
    # inside a simulation-box seam (Buzzard: z_true in [0.33, 0.37]) need
    # the same interval excised from the model's true-z integration.
    # Inactive unless zt_excl_high > zt_excl_low.
    excl_lo = _read_scalar_or_first(options, option_section,
                                    'zt_excl_low', 0.0)
    excl_hi = _read_scalar_or_first(options, option_section,
                                    'zt_excl_high', -1.0)
    if excl_hi > excl_lo:
        cfg['seam_weight'] = _seam_weight(cfg['z_grid'], excl_lo, excl_hi)
        print(f"[sel_function] true-z exclusion active: "
              f"[{excl_lo}, {excl_hi}]", flush=True)
    else:
        cfg['seam_weight'] = None

    # Per-module-instance cache for the plob EMG splines (see
    # _make_plob_splines) — plob_ltr_params never varies across a run in
    # production, so this turns an every-sample rebuild into a one-time
    # build reused for the rest of the chain.
    cfg['_plob_cache'] = {}

    n = cfg['lam_min'].size
    for key in ('lam_max', 'zob_min', 'zob_max', 'sigma_z'):
        if cfg[key].size != n:
            raise ValueError(
                f"sel_function: axis '{key}' size {cfg[key].size} != {n}")
    cfg['n_bins'] = n
    return cfg


# --- Plob spline helpers (linear in z, flat extrapolation past edges) ------

def _make_plob_splines(block, cache=None):
    """Return the 8 EMG coefficient splines over z, cached across samples.

    Prefers the datablock section ``plob_ltr_params/*`` if a publisher
    populated it (e.g. the ``prj_params`` cosmosis shim); otherwise
    falls back to the canonical in-code table from
    :class:`cosmology.prj_params.PrjParams`.

    ``plob_ltr_params`` is republished unconditionally every sample by
    the ``prj_params`` shim (nothing samples it), so rebuilding 8
    ``InterpolatedUnivariateSpline`` objects from scratch every
    ``execute()`` call is pure waste. Cache them in ``cache`` (the
    per-module-instance dict threaded through from ``setup()``),
    keyed on a cheap fingerprint of the raw arrays so the cache still
    self-invalidates correctly if the published values ever do change.
    """
    # ``cache=None`` preserves the historical one-argument reference API.
    # CosmoSIS passes the setup-owned cache; offline callers simply get a
    # correctly typed, uncached result.
    if cache is None:
        cache = {}
    source = (
        block
        if isinstance(block, (dm.DataBlockSource, dm.DumpSource))
        else dm.DataBlockSource(block)
    )
    try:
        keys = ('z',) + COEFF_NAMES
        fingerprint = tuple(
            np.asarray(source.array('plob_ltr_params', k), dtype=float).tobytes()
            for k in keys)
    except Exception:
        fingerprint = None

    if 'fingerprint' not in cache or cache['fingerprint'] != fingerprint:
        try:
            params = PrjParams.from_source(source)
        except Exception:
            params = PrjParams.default()
        cache['fingerprint'] = fingerprint
        cache['splines'] = params.splines()
    return cache['splines']


def _read_mor(block):
    """Return legacy MOR mapping via the shared HOD parameter normalizer."""
    source = (
        block
        if isinstance(block, (dm.DataBlockSource, dm.DumpSource))
        else dm.DataBlockSource(block)
    )
    parameters = dm.HODParameters.from_source(source)
    return {
        name: getattr(parameters, name)
        for name in (
            'log10_Mmin', 'log10_M1', 'alpha', 'epsilon',
            'sigma_lambda', 'z_pivot',
        )
    }


# --- Numba-fused variant of _compute_lam_nodes_and_P_HOD --------------------
#
# mu_sat/lcentral/delta/nu/degenerate depend only on (lnM, z) -- shape
# (n_lnm, n_z) -- but a fully vectorized P_HOD call broadcasts
# them across the full (n_lnm, n_z, N_q) ltr tensor and pays for two
# expensive per-element special-function calls there:
#   1. ``fallback`` (a narrow-Gaussian collapse for mu_sat ~ 0) is computed
#      UNCONDITIONALLY over the full tensor via one exp() call, even though
#      that branch (``tiny``) only depends on (lnM, z) and empirically never
#      fires across the sampled cluster_mor prior box (checked over 60
#      random draws: fires on 0.0% of (lnM, z) cells every time) -- the
#      whole computation was pure waste.
#   2. ``gammaln(x_safe)`` + ``exp(log_P)`` are both computed over the full
#      tensor even for cells the code's own final override already forces
#      to exactly 0.0 (``ltr < 0`` or ``ltr < lambda_central``) -- this
#      condition is cheap (no special functions) and empirically true for
#      ~8% of (lnM, z, q) points on average (up to ~20% for some draws),
#      so skipping it first avoids that fraction of gammaln/exp calls
#      entirely.
# A per-(lnM, z) outer loop computes the cell-level quantities once and
# reuses them across all N_q lambda-quadrature nodes in an inner loop,
# checking the exact-zero conditions before ever calling lgamma/exp.
# Validated against the numpy reference across 60 random cluster_mor draws:
# lam_k/W_k/degenerate match exactly; P_Mz matches to ~1e-10 (floating-point
# operation-order noise, not an approximation).
@njit(cache=True, error_model='numpy')
def _compute_lam_nodes_and_P_HOD_nb(lnM, z, log10_Mmin, log10_M1, alpha, epsilon,
                                     sigma_lambda, z_pivot, gl_t, gl_w, L,
                                     lam_k, W_k, P_Mz, degenerate):
    n_lnm = lnM.shape[0]
    n_z = z.shape[0]
    n_q = gl_t.shape[0]
    Mmin = 10.0 ** log10_Mmin
    M1 = 10.0 ** log10_M1
    dM1 = M1 - Mmin
    for k in range(n_lnm):
        M = math.exp(lnM[k])
        lcentral = 1.0 if M >= Mmin else 0.0
        dM = M - Mmin
        if dM < 0.0:
            dM = 0.0
        for zi in range(n_z):
            zz = z[zi]
            if dM1 <= 0.0 or dM <= 0.0:
                mu_sat = 0.0
            else:
                base = dM / dM1
                mu_sat = base ** alpha * ((1.0 + zz) / (1.0 + z_pivot)) ** epsilon

            delta = (sigma_lambda * mu_sat) ** 2
            nu = mu_sat + delta
            mu_eff = lcentral + mu_sat
            sig_eff = math.sqrt(nu) if nu > 0.0 else 0.0

            a_lo = mu_eff - L * sig_eff
            if a_lo < 0.0:
                a_lo = 0.0
            b_hi = mu_eff + L * sig_eff
            is_degenerate = b_hi <= a_lo
            degenerate[k, zi] = is_degenerate
            if is_degenerate:
                for qi in range(n_q):
                    lam_k[k, zi, qi] = 0.0
                    W_k[k, zi, qi] = 0.0
                    P_Mz[k, zi, qi] = 0.0
                continue

            half = 0.5 * (b_hi - a_lo)
            mid = 0.5 * (a_lo + b_hi)
            tiny = mu_sat <= POISSON_TOL
            log_nu = math.log(nu) if nu > 1e-300 else math.log(1e-300)

            for qi in range(n_q):
                ltr = mid + half * gl_t[qi]
                lam_k[k, zi, qi] = ltr
                W_k[k, zi, qi] = half * gl_w[qi]

                if ltr < 0.0 or ltr < lcentral:
                    P_Mz[k, zi, qi] = 0.0
                    continue
                if tiny:
                    dx = (ltr - lcentral) / FALLBACK_SIGMA
                    P_Mz[k, zi, qi] = math.exp(-0.5 * dx * dx) / (SQRT2PI * FALLBACK_SIGMA)
                    continue
                x = ltr - lcentral + delta
                if x <= 0.0:
                    P_Mz[k, zi, qi] = 0.0
                    continue
                # Continuous shifted-Poisson HOD density at this GL node:
                #   P = exp[-nu + (x - 1) log(nu) - log Gamma(x)]
                # ``math.lgamma`` is log Gamma(x), used directly for stable
                # evaluation instead of calculating Gamma(x) itself.
                log_P = -nu + (x - 1.0) * log_nu - math.lgamma(x)
                P_Mz[k, zi, qi] = math.exp(log_P)


def _compute_lam_nodes_and_P_HOD(lnM, z, mor, gl_t, gl_w, L=6.0):
    """Use the maintained fused NumPy/Numba quadrature execution path.

    The scalar model is owned by :class:`dm.PHOD`; this function retains the
    established hot kernel because it computes the cell-level HOD quantities
    once and reuses them across all true richness nodes. The kernel is
    validated against ``PHOD.make_ltr_quadrature`` by the selection tests.
    """
    n_lnm, n_z, n_q = lnM.size, z.size, gl_t.size
    lam_k = np.empty((n_lnm, n_z, n_q), dtype=np.float64)
    W_k = np.empty((n_lnm, n_z, n_q), dtype=np.float64)
    P_Mz = np.empty((n_lnm, n_z, n_q), dtype=np.float64)
    degenerate = np.empty((n_lnm, n_z), dtype=np.bool_)
    # The Numba function receives scalar fields because it cannot consume a
    # Python dataclass. The model normalization still happens once, here, at
    # the boundary; callers do not need to rebuild a MOR dictionary.
    parameters = _as_phod(mor).parameters
    _compute_lam_nodes_and_P_HOD_nb(
        np.asarray(lnM, dtype=np.float64), np.asarray(z, dtype=np.float64),
        parameters.log10_Mmin, parameters.log10_M1, parameters.alpha,
        parameters.epsilon, parameters.sigma_lambda, parameters.z_pivot,
        np.asarray(gl_t, dtype=np.float64), np.asarray(gl_w, dtype=np.float64),
        float(L), lam_k, W_k, P_Mz, degenerate)
    return lam_k, W_k, P_Mz, degenerate


def _unique_edges(lam_min, lam_max):
    """Sorted unique bin edges from per-bin (lam_min, lam_max)."""
    return np.unique(np.concatenate([lam_min, lam_max]))


def execute(block, config):
    import time
    t_start = time.perf_counter()

    source = dm.DataBlockSource(block)
    phod = dm.PHOD.from_source(source)
    plob = _make_plob_splines(block, config['_plob_cache'])

    n_bins = config['n_bins']

    # Single shared (lnM, z) grid across all bins → compute P_HOD and the
    # GL bracket ONCE. Individual bins hard-zero outside their z-window in
    # S_j. The S_i static cache covers the whole z-range already. The grid
    # itself is built once in setup() (static envelope/size options only).
    lnm_grid = config['lnm_grid']
    z_grid   = config['z_grid']
    t_phod = time.perf_counter()
    # Keep the fused hot path: PHOD supplies the normalized shared
    # parameters, while the established Numba kernel performs the large
    # (lnM, z, lambda_true) contraction without materializing extra arrays.
    lam_k, W_k, P_Mz, degenerate = _compute_lam_nodes_and_P_HOD(
        lnm_grid, z_grid, phod, config['gl_t'], config['gl_w'],
        config['L_lam'])
    dt_phod_ms = 1000.0 * (time.perf_counter() - t_phod)

    # S_i(lam_k, z) — closed form via CDF differencing at the unique bin
    # edges. plob_ltr_params are frozen Y3 splines, so mu/sigma/tau/fprj
    # evaluation on (lam_k, z) is pure numpy broadcasting; differencing
    # 5 edge CDFs gives the 4 unique S_i tables in one shot.
    #
    # Pass the 1-D z_grid (not a pre-broadcast (n_lnm, n_z, n_q) tensor)
    # so _plob_params can evaluate the 8 splines on 64 z-nodes only and
    # broadcast internally.  This single change saves ~130 ms/sample on
    # the (256, 64, 32) grid.
    n_lnm = lnm_grid.size
    n_z   = z_grid.size

    # work_dtype=float32: sigma/tau/fprj come out of transcendental
    # passes whose float32 error (7.9e-7 max relative on S_i, where S_i
    # > 1e-3 of peak) is the same order as the erfcx/Phi polynomials
    # already used downstream, for ~7.4 ms/sample. mu stays float64
    # inside _plob_params either way. Drop the kwarg to get bit-exact
    # float64 back.
    mu_p, sig_p, tau_p, fprj_p = _plob_params(lam_k, z_grid, plob,
                                              work_dtype=np.float32)
    # mu_p, sig_p, tau_p, fprj_p are already broadcast to the full
    # (n_lnm, n_z, n_q) tensor via the mu = a_mu + b_mu * ltr line inside
    # _plob_params (with ltr = lam_k).
    # The selection module owns the lambda partition.  Publish it so bsel
    # and all offline consumers use the same edges instead of maintaining a
    # second hard-coded copy.
    edges = _unique_edges(config['lam_min'], config['lam_max'])
    block['sel_function', 'lambda_edges'] = np.asarray(edges, dtype=float)
    block['sel_function', 'lambda_centres'] = 0.5 * (edges[:-1] + edges[1:])
    block['sel_function', 'zob_min'] = np.asarray(
        config['zob_min'], dtype=float)
    block['sel_function', 'zob_max'] = np.asarray(
        config['zob_max'], dtype=float)
    cdfs_at_edge = _cdf_lob_stacked(edges, mu_p, sig_p, tau_p, fprj_p)

    # First contract the PHOD × PLOB_LTR integrand over true richness for
    # every unique lambda edge. For one edge e, the quantity is
    #
    #   E_e(M,z) = integral[d ltr * P_HOD(ltr|M,z)
    #                         * F_PLOB(edge_e|ltr,z)].
    #
    # The result is a 2-D (lnM, z) surface per edge. A configured observed
    # richness bin is then just E_hi - E_lo, so no second observed-richness
    # quadrature or per-bin contraction is needed.
    weighted_hod = W_k * P_Mz
    edge_integrals = np.empty((edges.size, n_lnm, n_z), dtype=np.float64)
    for edge_index, cdf in enumerate(cdfs_at_edge):
        edge_integrals[edge_index] = np.sum(
            weighted_hod * cdf, axis=-1)
    edge_integrals = np.where(
        degenerate[None, :, :], 0.0, edge_integrals)

    lower_edge = np.searchsorted(edges, config['lam_min'])
    upper_edge = np.searchsorted(edges, config['lam_max'])
    richness_selection = (
        edge_integrals[upper_edge] - edge_integrals[lower_edge]
    )

    # Apply the independent Gaussian photo-z factor to every bin in one
    # broadcast. The final datavector layout is (bin, z, lnM).
    redshift_selection = _S_j(
        z_grid[None, :],
        config['zob_min'][:, None],
        config['zob_max'][:, None],
        config['sigma_z'][:, None],
    )
    # True-z seam excision (issue #8): the weight multiplies every bin's
    # S_j on the shared true-z grid, so all S_stack consumers inherit it.
    if config.get('seam_weight') is not None:
        redshift_selection = redshift_selection * config['seam_weight'][None, :]
    S_pack = np.ascontiguousarray(
        (richness_selection * redshift_selection[:, None, :])
        .transpose(0, 2, 1)
    )

    block['sel_function', 'lnM'] = lnm_grid
    block['sel_function', 'z']   = z_grid
    block['sel_function', 'S_stack'] = np.ascontiguousarray(S_pack)

    dt_ms = 1000.0 * (time.perf_counter() - t_start)
    print(f"[sel_function] {n_bins} S_ij tables ({config['n_lnm']}x"
          f"{config['n_z_shared']}) — P_HOD {dt_phod_ms:.0f} ms, "
          f"total {dt_ms:.0f} ms", flush=True)
    return 0


def cleanup(config):
    return 0
