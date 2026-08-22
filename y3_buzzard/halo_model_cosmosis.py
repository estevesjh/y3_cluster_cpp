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

from haloModel import biasModel, lensingModel, scaleShiftCosmo

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

    # one_halo_z_density: redshift at which the 1-halo DENSITY normalisation is
    # evaluated. Default 0.0 = the frozen COMOVING rho_m0 (the pipeline default;
    # first_halo_term(z=0)). Set >0 to use the PHYSICAL mean density
    # rho_m(z)=rho_m0(1+z)^3 in the 1-halo (issue #22). The CONCENTRATION stays
    # fixed (pre-set from one_halo_z + concentration_amplitude), so this isolates
    # the (1+z)^3 density factor. Evaluate one z-bin per run and stitch.
    try:
        z_density = float(options.get_double(section, "one_halo_z_density",
                                             default=0.0))
    except Exception:
        z_density = 0.0

    params_out = (R_perp_min, R_perp_max, R_perp_bins,
                  Radii_min, Radii_max, Radii_bins,
                  M_min, M_max, M_bins,
                  compute_lensing_1h, compute_lensing_2h, one_halo_z, c_amp,
                  z_density)
    return params_out
    

def execute(block, config):
    section_name = "haloModel"

    (R_perp_min, R_perp_max, R_perp_bins,
     Radii_min, Radii_max, Radii_bins,
     M_min, M_max, M_bins,
     compute_lensing_1h, compute_lensing_2h, one_halo_z, c_amp,
     z_density) = config

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
    R_perp = np.logspace(np.log10(R_perp_min), np.log10(R_perp_max), R_perp_bins)
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
    block[section_name, "bias"] = Bias
    # xi_NL(r, z) table for the b_sel_marg / sigma_prj integrands
    block["xi_nl", "r"]     = r_xi
    block["xi_nl", "z"]     = z
    block["xi_nl", "xi_nl"] = xi_NL

    # ----- lensing section: optional ----------------------------------------
    # Split into two independent branches: 1h (cheap analytic NFW
    # tables) and 2h (expensive cluster_toolkit Hankel transforms).
    # The modern red_shear_prj + bsel pipeline does not read either;
    # ReducedShear1hSel needs the 1h half; ReducedShearTotSel needs
    # both.  Skipping the 2h alone saves ~200-300 ms/sample.
    if compute_lensing_1h or compute_lensing_2h:
        lensModel = lensingModel(R_perp, omega_m, 200)
        scaleShift, hubbleShift = scaleShiftCosmo(z, cosmology)
        block[section_name, "r_sigma"] = R_perp
        block[section_name, "scale_shift"] = scaleShift
        block[section_name, "hubble_shift"] = hubbleShift
        block[section_name, "k"] = k_h

    if compute_lensing_1h:
        # one_halo_z enters ONLY the Child18 concentration (the issue-#3
        # defect). The density normalisation must stay the COMOVING
        # rho_m0: first_halo_term scales rho by (1+z)^3 (physical), and
        # the published tables are consumed as comoving -- evaluating the
        # whole term at z>0 would inflate DSigma by up to (1+z)^2
        # (measured +65% at the innermost radius for z=0.46). Setting
        # self.c first makes first_halo_term skip its own z=0 recompute.
        lensModel.concentration_at_M(M, z=one_halo_z,
                                     model_name="Child18")
        # Buzzard concentration boost: scale c BEFORE first_halo_term (which
        # reuses the pre-set lensModel.c) so the factor lands on the 1-halo
        # term and the published concentration alike.  c_amp=1.0 => Child18.
        if c_amp != 1.0:
            lensModel.c = lensModel.c * c_amp
        # z_density=0 -> comoving rho_m0 (default); z_density>0 -> physical
        # rho_m(z)=rho_m0(1+z)^3 (concentration stays fixed, pre-set above).
        lensModel.first_halo_term(M, z=z_density, conc_model_name="Child18")
        block[section_name, "Sigma_nfw"]  = lensModel.Sigma['1h']
        block[section_name, "dSigma_nfw"] = lensModel.dSigma['1h']
        block[section_name, "concentration"] = lensModel.c

    if compute_lensing_2h:
        lensModel.second_halo_term(z, k_nl, P_k_nl)
        # Wp_hh (xi_2halo on the R_perp grid, despite the Wp name) and its
        # mislabeled "Rp" axis are no longer published: unsupported, and
        # their only reader was the broken legacy wp_cluster.cuh
        # interpolation (wrong radial axis). The internal xi stays on
        # lensModel.Wp for validation; consumers needing xi read xi_nl.
        block[section_name, "Sigma_hh"]  = lensModel.Sigma['2h']
        block[section_name, "dSigma_hh"] = lensModel.dSigma['2h']

    return 0

def cleanup(config):
    pass