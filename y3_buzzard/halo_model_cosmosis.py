import os
import numpy as np
from cosmosis.datablock import option_section, names

from scipy.interpolate import interp1d
from scipy.special import erf
import cluster_toolkit as ct

# ---- Disable GSL's abort-on-error handler --------------------------------
# cluster_toolkit links libgsl (libgsl.so.25 in the y3cl_je conda env).
# At extreme cosmologies (e.g. high log10As, low Omega_m) GSL's adaptive
# qag integrator inside cluster_toolkit.peak_height.nu_at_M can hit
#   "gsl: qag.c:247: ERROR: roundoff error prevents tolerance from being achieved"
# and the default GSL handler calls abort(), killing the worker process.
# Replace it with the no-op handler (NULL pointer) so GSL routines just
# return non-zero status codes; cluster_toolkit then propagates a Python
# exception instead of aborting.  Apriori/emcee samplers can then
# gracefully assign logL = -inf to the offending draw.
def _disable_gsl_abort_handler() -> None:
    import ctypes
    try:
        # cluster_toolkit's linked libgsl.  Use ctypes.util.find_library
        # as fallback if the cffi handle isn't introspectable.
        import ctypes.util
        lib_path = ctypes.util.find_library("gsl")
        if lib_path is None:
            # Fallback: rely on cluster_toolkit having loaded libgsl already.
            lib_path = "libgsl.so.25"
        gsl = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        gsl.gsl_set_error_handler_off.restype = ctypes.c_void_p
        gsl.gsl_set_error_handler_off()
        print("[halo_model_cosmosis] GSL abort-on-error handler disabled "
              f"(via {lib_path})", flush=True)
    except OSError as exc:
        print(f"[halo_model_cosmosis] WARNING: could not disable GSL "
              f"abort handler: {exc}", flush=True)

_disable_gsl_abort_handler()

from astropy.constants import G

from haloModel import biasModel, lensingModel, scaleShiftCosmo, max1h2hOperators, ct_2hTerm

##################################################################
############# Build Wp(R,z) and Gammat(R,z) ############
# Estimates Halo Model Parameters
# Wp, \Sigma, \Delta \Sigma and bias(z)
# 
#############################################
# Author: Johnny Esteves
# Created: May 16, 2024
#############################################

import astropy.cosmology
cosmo_names = names.cosmological_parameters

def setup(options):
    section = option_section
    #Mpc/h comoving distance, distance on the sky
    R_perp_min = options[section,"R_perp_min"]
    R_perp_max = options[section,"R_perp_max"]
    R_perp_bins = int(options[section,"R_perp_bins"])
    # Optional sparse core segment below R_perp_min: R_perp_core_bins knots
    # on [R_perp_core, R_perp_min), log-spaced, prepended to the main grid
    # (whose knots stay unchanged).  These radii are not fitted; they carry
    # the profile down to the core radius r_sigma[0] = R_perp_core, inside
    # which m2d_core holds the projected mass (miscentering needs both).
    # 8 knots on [1e-3, 0.1) reach the 1.4e-5 floor of the gamma-miscentred
    # DeltaSigma on R >= 0.2 (analytic NFW); 4 give 1.2e-4.
    R_perp_core = (options[section, "R_perp_core"]
                   if options.has_value(section, "R_perp_core") else None)
    R_perp_core_bins = (int(options[section, "R_perp_core_bins"])
                        if options.has_value(section, "R_perp_core_bins") else 8)
    R_perp = np.logspace(np.log10(R_perp_min), np.log10(R_perp_max), R_perp_bins)
    if R_perp_core is not None:
        if not (0.0 < R_perp_core < R_perp_min) or R_perp_core_bins < 1:
            raise ValueError("halo_model: needs 0 < R_perp_core < R_perp_min "
                             "and R_perp_core_bins >= 1")
        R_perp = np.concatenate([
            np.geomspace(R_perp_core, R_perp_min, R_perp_core_bins, endpoint=False),
            R_perp])

    # radii of xi_hm, in Mpc/h
    Radii_min = options[section,"Radii_min"]
    Radii_max = options[section,"Radii_max"]
    Radii_bins = int(options[section,"Radii_bins"])

    #mass (float): Halo mass Msun/h.
    M_min = options[section,"M_min"]
    M_max = options[section,"M_max"]
    M_bins = int(options[section,"M_bins"])

    # Two independent toggles for the lensing stack:
    #
    #   compute_lensing_1h  publish Sigma_nfw, dSigma_nfw, concentration
    #                       (cluster_toolkit analytic NFW tables, ~cheap).
    #                       Read by the legacy Sigma1hSel / DSigma1hSel
    #                       / Shear1hSel / ReducedShear1hSel modules.
    #
    #   compute_lensing_2h  publish Sigma_hh, dSigma_hh, Wp_hh
    #                       (cluster_toolkit ct_2hTerm: xi_mm + Hankel
    #                       Sigma_at_R + DeltaSigma_at_R — ~200-300 ms
    #                       and the dominant cost of this module).
    #                       Read by the *TotSel variants only.
    #
    # Costanzi-2026 projection pipeline (red_shear_prj + bsel) only
    # needs haloModel/bias and xi_nl/xi_nl: set both toggles to F.
    # ReducedShear1hSel alone: 1h=T, 2h=F.  The legacy Tot stack: both T.
    #
    # Backward-compat: the old single knob `compute_lensing` controls
    # both when set (maps to 1h AND 2h); defaults are (1h=T, 2h=T).
    def _bool(key, default):
        try:
            return bool(options.get_bool(section, key, default=default))
        except Exception:
            return default
    compute_lensing = _bool("compute_lensing", True)
    compute_lensing_1h = _bool("compute_lensing_1h", compute_lensing)
    compute_lensing_2h = _bool("compute_lensing_2h", compute_lensing)
    # Option for max1h2h piecewise approximation (supports compute_max1h2h or max1h2hOperator)
    compute_max1h2h = _bool("compute_max1h2h", False) or _bool("max1h2hOperator", False) or _bool("max1h2hOperators", False)

    # z_halo: the FIXED redshift at which the 1h concentration is
    # evaluated -- the owner-ratified convention (issue #3, review
    # 2026-08-20): the 1h term uses c(M, z_halo) with z_halo an ini
    # parameter, default 0.4 (~the survey mean); per-z tables are
    # deliberately NOT the model. Density normalisation stays the
    # comoving rho_m0 regardless (see execute()).
    #
    # Legacy runs (widePlanck self-closure DVs, the frozen fiducial
    # dumps behind the hard-coded cross-backend pins) were generated at
    # z=0: those inis now pin `z_halo = 0.0` explicitly. `one_halo_z`
    # is honored as a deprecated alias when `z_halo` is absent.
    try:
        one_halo_z = float(options.get_double(section, "z_halo",
                                              default=np.nan))
    except Exception:
        one_halo_z = np.nan
    if not np.isfinite(one_halo_z):
        try:
            one_halo_z = float(options.get_double(section, "one_halo_z",
                                                  default=0.4))
        except Exception:
            one_halo_z = 0.4

    # concentration_amplitude: multiply the Child18 c(M, z_halo) by this
    # factor.  Buzzard clusters are ~1.25x more concentrated than Child18
    # (measured c = R_200m/r_s vs the Child18 relation, median ratio ~1.25),
    # which is the small-R 1-halo DeltaSigma deficit.  Applied consistently to
    # BOTH the 1-halo term (via lensModel.c, reused by first_halo_term) and the
    # published concentration (consumed by the miscentered NFW through
    # set_concentration_table).  Default 1.0 (Child18 unchanged).
    try:
        c_amp = float(options.get_double(section, "concentration_amplitude",
                                         default=1.0))
    except Exception:
        c_amp = 1.0

    # concentration_fixed: a single, mass-independent NFW concentration
    # c = R_200m / r_s that REPLACES the c(M, z_halo) relation (conc_model,
    # z_halo and concentration_amplitude are then unused).  Like the
    # amplitude, it is set on lensModel.c before first_halo_term, so it
    # lands on the 1-halo tables, the published `concentration` (miscentred
    # NFW, the gauss_legendre xi_nfw) and the max1h2h r_trans/c0/gap terms
    # alike.  Unset (default) keeps the c(M) relation.  The Buzzard runs use
    # concentration_fixed = 5.0 (des-cluster-nersc be6757b).
    try:
        c_fixed = float(options.get_double(section, "concentration_fixed",
                                           default=np.nan))
    except Exception:
        c_fixed = np.nan
    if np.isfinite(c_fixed) and not c_fixed > 0.0:
        raise ValueError("halo_model: concentration_fixed must be > 0")

    # conc_model: mass-concentration relation for the 1-halo term and the
    # published `concentration` (haloModel.concentration_at_M's
    # model_name). Default "Child18" (pipeline default, unchanged);
    # "Duffy08" is the other supported option (haloModel.py).
    try:
        conc_model = options.get_string(section, "conc_model",
                                        default="Child18")
    except Exception:
        conc_model = "Child18"

    # one_halo_z_density: redshift at which the 1-halo DENSITY normalisation is
    # evaluated. Default 0.0 = the frozen COMOVING rho_m0 (the pipeline default;
    # first_halo_term(z=0)). Set >0 to use the PHYSICAL mean density
    # rho_m(z)=rho_m0(1+z)^3 in the 1-halo (issue #22). The CONCENTRATION stays
    # fixed (pre-set from one_halo_z + concentration_amplitude), so this isolates
    # the (1+z)^3 density factor. Evaluate one z-bin per run and stitch.
    # *** APPROXIMATION (issue #22, J.Esteves): this puts (1+z)^3 into the analytic
    # NFW that fills the z-FREE (R,M) dSigma_nfw table, so per stitched z-bin the
    # (1+z)^3 lands OUTSIDE the C++ z-integral (Wb weights), at the bin centre --
    # NOT inside the integrand as it should be. Rigorous fix = 3-D (R,M,z) NFW table
    # + phi(R,M,z) evaluated inside the SelGLCore z-integral. Use for ~5-15% checks.
    try:
        z_density = float(options.get_double(section, "one_halo_z_density",
                                             default=0.0))
    except Exception:
        z_density = 0.0

    # one_halo_physical_density: the RIGOROUS in-integrand treatment of the
    # physical mean density rho_m(z) = rho_m0 (1+z)^3 (2026-08-24 decision,
    # resolves the issue-#22 approximation above for the shear evaluators).
    # It uses the exact fixed-c NFW identity
    #     DSigma_phys(R | z) = (1+z)^2 * DSigma_com(R (1+z)),
    # i.e. the boundary shrink r_s -> r_s/(1+z) (the "c(1+z)" rescale) plus
    # the net (1+z)^2 amplitude. Consumers apply it as: (1+z)^2 folded into
    # the shear z-weight (exact per z node) + the query-radius rescale
    # (exact where z is live in the integrand; at the bin's
    # selection-weighted z_eff in the z-contracted 0d evaluators).
    # NUMBER COUNTS ARE NEVER TOUCHED. Requires z_density = 0 (comoving
    # tables) -- combining both would double count the density evolution.
    try:
        physical_density = bool(options.get_bool(
            section, "one_halo_physical_density", default=False))
    except Exception:
        physical_density = False
    if physical_density and z_density != 0.0:
        raise ValueError(
            "halo_model: one_halo_physical_density=T requires "
            "one_halo_z_density = 0 (comoving tables); combining both "
            "double-counts the (1+z)^3 density evolution")

    params_out = (R_perp,
                  Radii_min, Radii_max, Radii_bins,
                  M_min, M_max, M_bins,
                  compute_lensing_1h, compute_lensing_2h, one_halo_z, c_amp,
                  z_density, physical_density, conc_model, compute_max1h2h,
                  c_fixed)
    return params_out


def execute(block, config):
    section_name = "haloModel"

    (R_perp,
     Radii_min, Radii_max, Radii_bins,
     M_min, M_max, M_bins,
     compute_lensing_1h, compute_lensing_2h, one_halo_z, c_amp,
     z_density, physical_density, conc_model, compute_max1h2h,
     c_fixed) = config

    # cosmo parameters
    omega_m = block[cosmo_names, "omega_m"]
    omega_b = block[cosmo_names, "omega_b"]
    H0 = block[cosmo_names, "H0"]

    cosmology = astropy.cosmology.FlatLambdaCDM(H0*100, omega_m, Ob0=omega_b, Tcmb0=2.725)

    # load camb matter-matter power spectrums
    # linear is used for the bias
    k_h = block["matter_power_lin", "k_h"]
    P_k = block["matter_power_lin", "p_k"]
    z_k = block["matter_power_lin", "z"]

    # nonlinear P(k) -- fall back to linear if cp_camb's nonlinear emulator
    # is not plugged in (wiring test only; physics approximate).
    if block.has_section("matter_power_nl"):
        P_k_nl = block["matter_power_nl", "p_k"]
        k_nl   = block["matter_power_nl", "k_h"]
        z_nl   = block["matter_power_nl", "z"]
    else:
        P_k_nl, k_nl, z_nl = P_k, k_h, z_k

    z = z_k

    if block.has_section("growth_parameters"):
        z_az = block["growth_parameters", "z"]
        daz  = block["growth_parameters", "d_z"]
        # CosmoSIS publishes an un-normalised growth factor
        # (D -> a at early times, so D(0) != 1).  bias(M, z) needs the
        # ratio sigma(M, z)/sigma(M, 0) = D(z)/D(0), so renormalise here.
        daz  = np.interp(z, z_az, daz) / np.interp(0.0, z_az, daz)
    else:
        # cp_camb mode: no growth_parameters.  Approximate D(z) via
        # the shape of sigma_8 only (daz = 1 at z=0, drops slowly).
        # For a pure-wiring-timing test, just set D(z)=1 and move on;
        # Bias(M, z) is wrong but the timing measurement is correct.
        daz = np.ones_like(z)

    # compute overdensities; physical not comoving
    rho_c0 = float(cosmology.critical_density0.to('Msun/Mpc^3').value)
    # rho_cz = cosmology.critical_density(z).to('Msun/Mpc^3').value
    rho_m = omega_m*rho_c0
    rho_mz = rho_m*(1.+z)**3
    
    # setup bins
    nz = len(z)
    Radii = np.logspace(np.log10(Radii_min), np.log10(Radii_max), Radii_bins)
    M = np.logspace(np.log10(M_min), np.log10(M_max), M_bins)
    logM = np.log(M)

    # Compute Bias (M, Z)
    # The bias is computed at the peak height (z=0) from cluster toolkit
    # The peak-height evolution is computed from the growth function D(z)
    # The vector Bias has shape (z.size, M.size)
    bM = biasModel(k_h, P_k[0], omega_m, odelta=200)
    nu = bM.nu_at_M(M)
    Bias = np.array([bM.bias_at_nu(nu/dazi) for dazi in daz])

    # xi_NL(r, z) pre-tabulated on a fixed log-r grid for the C++
    # b_sel_marg / sigma_prj integrands. Evaluated per z slice; kept
    # outside the `compute_lensing` branch because the Costanzi-2026
    # projection pipeline always needs it.
    r_xi = np.logspace(-3.0, 3.0, 128)  # Mpc/h comoving
    xi_NL = np.zeros((z.size, r_xi.size))
    for iz, zz in enumerate(z):
        xi_NL[iz] = ct.xi.xi_mm_at_r(r_xi, k_nl, P_k_nl[iz])

    # ----- always-on datablock writes (consumed by bsel + red_shear_prj) -----
    block[section_name, "m_h"] = M
    block[section_name, "lnM"] = logM
    block[section_name, "z"] = z
    block[section_name, "rhoc"] = rho_mz
    # rho_m_ref: THE single NFW reference density every profile consumer
    # (CPU/CUDA mis readers, Python replicas, radial-series family) uses
    # for BOTH the halo boundary r_200 = [3M/(800 pi rho_m_ref)]^(1/3)
    # and the amplitude rho_s = delta_c * rho_m_ref. Identical to the
    # density first_halo_term builds the centred tables with:
    # rho_m0 * (1+z_density)^3 (comoving at the default z_density = 0,
    # physical when one_halo_z_density is set). Unified-convention
    # decision 2026-08-24 (supersedes the 200c/rho_crit mis-profile
    # hybrid).
    # h-unit convention: masses are Msun/h and radii Mpc/h, so rho_ref
    # must be the h-unit critical density (2.775e11 Msun h^2/Mpc^3) --
    # the SAME constant lensingModel.rhoc0 uses for the centred tables.
    # (execute()'s astropy rho_m above is PHYSICAL Msun/Mpc^3 and feeds
    # only the published rhoc(z) array.)
    RHOC_HUNITS = 2.77533742639e+11
    block[section_name, "rho_m_ref"] = (omega_m * RHOC_HUNITS
                                        * (1.0 + z_density) ** 3)
    rho_m_comoving = omega_m * RHOC_HUNITS
    m3d_hh = ct_2hTerm.m3d_at_r(r_xi, xi_NL, rho_m=rho_m_comoving)
    int_r2_xi_nl = ct_2hTerm.m3d_at_r(r_xi, xi_NL, rho_m=1.0)
    # 0/1 flag for the rigorous physical-density treatment (see setup);
    # every shear-profile consumer reads this SAME entry.
    block[section_name, "one_halo_physical_density"] = int(physical_density)
    block[section_name, "bias"] = Bias
    # xi_NL(r, z) table for the b_sel_marg / sigma_prj integrands
    block["xi_nl", "r"]            = r_xi
    block["xi_nl", "z"]            = z
    block["xi_nl", "xi_nl"]        = xi_NL
    block["xi_nl", "int_r2_xi_nl"] = int_r2_xi_nl
    block["xi_nl", "m3d_nl"]       = m3d_hh
    block[section_name, "m3d_hh"]  = m3d_hh

    # ----- lensing section: optional ----------------------------------------
    # Split into two independent branches: 1h (cheap analytic NFW
    # tables) and 2h (expensive cluster_toolkit Hankel transforms).
    # The modern red_shear_prj + bsel pipeline does not read either;
    # ReducedShear1hSel needs the 1h half; ReducedShearTotSel needs
    # both.  Skipping the 2h alone saves ~200-300 ms/sample.
    if compute_lensing_1h or compute_lensing_2h or compute_max1h2h:
        lensModel = lensingModel(R_perp, omega_m, 200)
        scaleShift, hubbleShift = scaleShiftCosmo(z, cosmology)
        block[section_name, "r_sigma"] = R_perp
        block[section_name, "scale_shift"] = scaleShift
        block[section_name, "hubble_shift"] = hubbleShift
        block[section_name, "k"] = k_h

    if compute_lensing_1h or compute_max1h2h:
        # one_halo_z enters ONLY the Child18 concentration (the issue-#3
        # defect). The density normalisation must stay the COMOVING
        # rho_m0: first_halo_term scales rho by (1+z)^3 (physical), and
        # the published tables are consumed as comoving -- evaluating the
        # whole term at z>0 would inflate DSigma by up to (1+z)^2
        # (measured +65% at the innermost radius for z=0.46). Setting
        # self.c first makes first_halo_term skip its own z=0 recompute.
        if np.isfinite(c_fixed):
            # Fixed, mass-independent concentration (concentration_fixed).
            lensModel.c = np.full(np.shape(M), c_fixed)
        else:
            lensModel.concentration_at_M(M, z=one_halo_z,
                                         model_name=conc_model)
            # Buzzard concentration boost: scale c BEFORE first_halo_term
            # (which reuses the pre-set lensModel.c) so the factor lands on
            # the 1-halo term and the published concentration alike.
            # c_amp=1.0 => no boost.
            if c_amp != 1.0:
                lensModel.c = lensModel.c * c_amp
        # z_density=0 -> comoving rho_m0 (default); z_density>0 -> physical
        # rho_m(z)=rho_m0(1+z)^3 (concentration stays fixed, pre-set above).
        lensModel.first_halo_term(M, z=z_density, conc_model_name=conc_model)
        block[section_name, "Sigma_nfw"]     = lensModel.Sigma['1h']
        block[section_name, "dSigma_nfw"]    = lensModel.dSigma['1h']
        block[section_name, "Sigma_bar_nfw"] = lensModel.sigma_bar['1h']
        block[section_name, "concentration"] = lensModel.c

    if compute_lensing_2h or compute_max1h2h:
        lensModel.second_halo_term(z, k_nl, P_k_nl)
        # Wp_hh (xi_2halo on the R_perp grid, despite the Wp name) and its
        # mislabeled "Rp" axis are no longer published: unsupported, and
        # their only reader was the broken legacy wp_cluster.cuh
        # interpolation (wrong radial axis). The internal xi stays on
        # lensModel.Wp for validation; consumers needing xi read xi_nl.
        block[section_name, "Sigma_hh"]     = lensModel.Sigma['2h']
        block[section_name, "dSigma_hh"]    = lensModel.dSigma['2h']
        block[section_name, "Sigma_bar_hh"] = lensModel.sigma_bar['2h']

    # ----- max(1h, 2h) piecewise approximation terms: optional ------------
    # The Complete Unified Profile:
    #
    # Because DeltaSigma(R) = barSigma(<R) - Sigma(R) and
    # d/dR [R^2 barSigma(<R)] = 2 R Sigma(R), enforcing quadratic matching
    # for DeltaSigma uniquely determines the exact unified profiles
    # for Sigma(R), barSigma(<R), and DeltaSigma(R):
    #
    # 1. Projected Surface Density Sigma(R):
    #   For R < r_trans(M, z):
    #     Sigma(R) = Sigma_1h(R, M) + Delta_gap^(Sigma)(M, z)
    #                + 2 * Delta_gap(M, z) * [1 - (R / r_trans(M, z))^2]
    #   For R >= r_trans(M, z):
    #     Sigma(R) = b(M, z) * Sigma_hh(R, z)
    #   where the boundary surface density gap is:
    #     Delta_gap^(Sigma)(M, z) = b(M, z) * Sigma_hh(r_trans, z) - Sigma_1h(r_trans, M)
    #
    # 2. Enclosed Mean Surface Density barSigma(<R):
    #   For R < r_trans(M, z):
    #     barSigma(<R) = barSigma_1h(<R, M) + Delta_gap^(Sigma)(M, z)
    #                    + 2 * Delta_gap(M, z) - Delta_gap(M, z) * (R / r_trans(M, z))^2
    #   For R >= r_trans(M, z):
    #     barSigma(<R) = b(M, z) * barSigma_hh(<R, z) + C_0(M, z) / R^2
    #
    # 3. Excess Surface Density DeltaSigma(R) = barSigma(<R) - Sigma(R):
    #   Subtracting Sigma(R) from barSigma(<R), the constant sheet
    #   Delta_gap^(Sigma) + 2 Delta_gap cancels out identically, leaving:
    #   For R < r_trans(M, z):
    #     DeltaSigma(R) = DeltaSigma_1h(R, M) + Delta_gap(M, z) * (R / r_trans(M, z))^2
    #   For R >= r_trans(M, z):
    #     DeltaSigma(R) = b(M, z) * DeltaSigma_hh(R, z) + C_0(M, z) / R^2
    #   where Delta_gap(M, z) = [b * DeltaSigma_hh(r_trans) + C_0 / r_trans^2] - DeltaSigma_1h(r_trans).
    #
    # Exact Identity: DeltaSigma_hm(R) == barSigma_hm(<R) - Sigma_hm(R) holds analytically everywhere.
    # Continuity: All three profiles (Sigma, barSigma, DeltaSigma) are strictly C^0 continuous at R = r_trans.
    # DataBlock inputs: All tables are strictly 2D arrays:
    #   Sigma_1h(R, M), Sigma_hh(R, z), r_trans(M, z), C_0(M, z), Delta_gap(M, z), Delta_gap^(Sigma)(M, z).
    if compute_max1h2h or (compute_lensing_1h and compute_lensing_2h):
        ops = max1h2hOperators(lensModel)
        r_trans = ops.compute_rtrans(M, z, Bias, r_xi, xi_NL)
        c0_trans = ops.compute_c0(r_trans, M, z, Bias, r_xi, m3d_hh)
        delta_gap_0 = ops.compute_delta_gap_0(r_trans, M, z, Bias, z_density=z_density)
        delta_gap_1 = ops.compute_delta_gap_1(r_trans, c0_trans, M, z, Bias, z_density=z_density)

        block[section_name, "r_trans"]     = r_trans
        block[section_name, "c0_trans"]    = c0_trans
        block[section_name, "delta_gap_0"] = delta_gap_0
        block[section_name, "delta_gap_1"] = delta_gap_1

        # Core mass: projected max-model mass inside the table floor
        # r_core = R_perp[0], from the unified barSigma above (r_core < r_trans):
        #   M_2d,core(M, z) = pi r_core^2 [barSigma_1h(<r_core, M) + Delta_gap^(Sigma)
        #                     + 2 Delta_gap - Delta_gap (r_core / r_trans)^2]   [Msun/h]
        # Profile consumers that need Sigma below the table (miscentering)
        # take the core as a uniform disk of this mass; keep r_sigma[0] small
        # (~1e-3 Mpc/h, via R_perp_core) so the disk is accurate (1e-5 at
        # 1e-3, 3e-2 at 0.1).
        r_core = R_perp[0]
        if np.any(r_core >= r_trans):
            raise ValueError("halo_model: r_sigma[0] must lie inside r_trans for m2d_core")
        sbar_core = (np.asarray(lensModel.sigma_bar['1h'])[:, 0][None, :]
                     + delta_gap_0 + 2.0 * delta_gap_1
                     - delta_gap_1 * (r_core / r_trans) ** 2)
        block[section_name, "m2d_core"] = np.pi * r_core**2 * sbar_core * 1.0e12

    return 0

def cleanup(config):
    pass