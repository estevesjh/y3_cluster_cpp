"""Halo-model ingredients for cluster lensing and halo bias.

This module provides the physical ingredients used by the cluster-observable
calculations:

* a one-halo NFW contribution to the projected surface density and excess
  surface density, using a chosen mass--concentration relation;
* a two-halo contribution obtained from the nonlinear matter power spectrum
  through ``cluster_toolkit``; and
* the Tinker et al. (2010) halo-bias relation, evaluated from the peak height
  of a halo in the linear matter power spectrum.

For a halo of mass ``M`` at redshift ``z``, ``lensingModel`` stores the
one-halo and two-halo contributions to ``Sigma`` and ``DeltaSigma`` as
separate channels. The two-halo channel also provides the projected
halo--matter correlation function. ``biasModel`` converts the linear power
spectrum into the mass-dependent halo bias used to weight the two-halo term.

The analytic NFW lensing expressions follow Wright & Brainerd (2000). Mass
and radius units, the halo mass definition, and the density normalization
must be kept consistent with the concentration relation and the consuming
observable pipeline.

The mass--concentration relations are implemented in ``concentration.py``.
The original implementation was written by Johnny Esteves on May 16, 2024.
"""
from pathlib import Path
import sys

import numpy as np
import cluster_toolkit as ct

# Dual-mode import: as a package member (``cosmology.halo_model``) via the
# src/pipelines sys.path convention, or loaded by file path as a CosmoSIS
# module (no package context), where the bootstrap below makes the absolute
# ``cosmology.*`` imports resolve.
_PIPELINES_DIR = str(Path(__file__).resolve().parents[1])
if _PIPELINES_DIR not in sys.path:
    sys.path.insert(0, _PIPELINES_DIR)

from cosmology.nfw_model import sigmaNFW_Analytical, deltaSigmaNFW_Analytical
from cosmology.concentration import child18_mass_concentration, duffy_concentration_relation


class lensingModel(object):
    """ Compute the lensing model for a given mass-concentration relation
        and bias model

        The lensing model is basically a halo model with the fist (1h) and second (2h) halos terms.

        The 1h term is a NFW profile (computed analytically, see: Wright & Brained 2000)
        The 2h term is based on a given power-spectrum (computed from cluster toolkit via two point correlation function)

        The outputs are \Delta\Sigma and \Sigma for a given mass M and redshift z.

        All the mass definitions are baseond the mean mass density (e.g. 200m)

        input:
            R : array, radius values

        Example:
        --------
        lensModel = lensingModel(R, omega_m=0.3, odelta=200)
        lensModel.first_halo_term(M, z=0, conc_model_name='Child18')
        lensModel.second_halo_term(z, k, P)

        bias = 3.
        dSigma = lensModel.dSigma['1h'] + bias*lensModel.dSigma['2h']
        Sigma  = lensModel.Sigma['1h'] + bias*lensModel.Sigma['2h']

        twoPointCorrelationFunction = lensModel.Wp
    """
    def __init__(self, R, omega_m=0.3, odelta=200, exclusion=True):
        self.omega_m = omega_m
        self.odelta = odelta
        self.exclusion = exclusion
        self.rhoc0 = 2.77533742639e+11  # Msun/Mpc^3/h^2 (critical density)
        self.rhom0 = self.omega_m * self.rhoc0
        self.R = R

        self.Sigma = {'1h':None, '2h':None}
        self.dSigma = {'1h':None, '2h':None}

    def first_halo_term(self, M, z=0.0, conc_model_name='Child18'):
        if not hasattr(self,'c'):
             self.concentration_at_M(M, z=z, model_name=conc_model_name)

        if isinstance(M, (int, float)):
            M = np.array([M])

        if isinstance(self.c, (int, float)):
            self.c = self.c * np.ones_like(M)

        # units are Msun/pc^2
        mm, rr  = np.meshgrid(M, self.R, indexing='ij')
        self.Sigma['1h'] = sigmaNFW_Analytical(rr, mm, self.c[:,np.newaxis], rho_c=self.rhom0*(1+z)**3)/1e12
        self.dSigma['1h'] = deltaSigmaNFW_Analytical(rr, mm, self.c[:,np.newaxis], rho_c=self.rhom0*(1+z)**3)/1e12

    def second_halo_term(self, z, k, P):
        ## Cluster Toolkit Halo-Halo projected \Sigma and \Delta \Sigma second halo term computation
        ## It also computed the 2-point correlation function \Xi_{mm}
        # The mass and concentration are dummy values
        # The redshift is needed if the power-spectrum is non linear
        p2h = ct_2hTerm(self.omega_m, Md=1e14, cd=5, bias=1.0)
        p2h.pk_to_dsigma(self.R, k, P, z)
        self.Sigma['2h'] = p2h.Sigma
        self.dSigma['2h'] = p2h.dSigma
        self.Wp = p2h.Xi

    def concentration_at_M(self, M, z=0.0, model_name="Child18", **kwargs):
        """ Set the mass-concentration model
        """
        if model_name=="Child18":
            self.c = child18_mass_concentration(M, z, halo_sample = 'stacked_nfw')

        elif model_name=='Duffy08':
            self.c = duffy_concentration_relation(M, z_eff=z)
        else:
            raise Exception('Unknown mass-concentration model, %s.' % (model_name))
        return self.c


class biasModel:
    """ Compute the Bias Tinker et al. 2010 model
        for a given linear power-spectrum

        The calculation is based on the peak height of a top-hat sphere
        of lagrangian radius R corresponding to a mass M of linear
        power-spectrum.

        This class call cluster_toolkit to compute the peak height
        https://cluster-toolkit.readthedocs.io/en/latest/

        input:
            k : array, wavenumbers
            P : array, linear power-spectrum
            z : array, redshift

        Example:
        --------
        bM = biasModel(k, P, omega_m=0.3)
        bias = bM.bias_at_M(M, odelta=200)
    """
    def __init__(self, k, P, omega_m=0.3, odelta=200):
        self.k = k
        self.P = P
        self.omega_m = omega_m
        self.odelta = odelta

    def bias_at_M(self, M):
        """ Compute the bias for a given mass M

        Based on Bias Tinker et al. 2010 Eqn 6

        Computes peak height of top hat sphere of lagrangian radius R [Mpc/h comoving]
        corresponding to a mass M [Msun/h] of linear power spectrum.
        """
        if not hasattr(self,'nu'):
            self.nu = self.compute_nu(M)

        bias = self.bias_at_nu(self.nu, odelta=self.odelta)
        return bias

    def nu_at_M(self, M):
        """ Compute peak-height
        https://cluster-toolkit.readthedocs.io/en/latest/api/cluster_toolkit.peak_height.html#cluster_toolkit.peak_height.nu_at_M

        """
        Nu0 = ct.peak_height.nu_at_M(M, self.k, self.P, self.omega_m)
        return Nu0

    def bias_at_nu(self,nu):
        """ Bias Tinker et a. 2010 Eqn 6
        """
        A, a, B, b, C, c = self.get_tinker_pars()
        bias = self._bias_at_nu(nu, A, a, B, b, C, c, deltac=1.686)
        return bias

    def get_tinker_pars(self):
        y = np.log10(self.odelta)
        # for delta=200
        tinker_best_fit = {
            'A': 1.0 + 0.24*y*np.exp(- (4/y)**4),
            'a': 0.44*y - 0.88,
            'B': 0.183,
            'b': 1.5,
            'C': 0.019+0.107*y+0.19*np.exp(-(4/y)**4),
            'c': 2.4
        }
        return [tinker_best_fit[col] for col in ['A','a','B','b','C','c']]

    def _bias_at_nu(self, nu, A, a, B, b, C, c, deltac=1.686):
        """ Bias Tinker et a. 2010 Eqn 6
        """
        res = 1.0 - A * nu**a/ (nu**a + deltac**a)
        res+= B * nu**b
        res+= C* nu**c
        return res

## Using Cluster Tool-kit
# The NSIZE of 50 is fast and has an accuracy of 0.1%
class ct_2hTerm(object):
    ## following cluster toolkit definition
    RHO_C = 2.77533742639e+11 # Msun/Mpc^3/h^2 (critical density)

    def __init__(self,omega_m=0.3,exclusion=True,NSIZE=50, Md=1e13, cd=4.,bias=1.00,
                 dsigma_method='direct'):
        self.omega_m = omega_m
        self.NSIZE = NSIZE
        self.Rfix = np.logspace(-3., 3., NSIZE, base=10) #Xi_hm MUST be evaluated to higher than BAO
        # Md/cd parameterize cluster_toolkit's inner-edge NFW extension in
        # Sigma_at_R (below Rfix[0]) and the 'sandwich' stabilizer; the
        # published table is the pure b=1 two-halo term under both methods.
        self.Md, self.cd = Md, cd
        self.bias = bias
        # 'exclusion' is a deprecated no-op kept for call-site
        # compatibility: the NFW add/subtract pair it used to gate is a
        # numerical stabilizer, not physics, and only enters the
        # 'sandwich' method (where skipping it is never correct).
        self.exclusion = exclusion
        if dsigma_method not in ('direct', 'sandwich'):
            raise ValueError("dsigma_method must be 'direct' or 'sandwich', "
                             "got %r" % (dsigma_method,))
        self.dsigma_method = dsigma_method

    def _pk_to_xi(self,k,pk):
        assert np.ndim(pk) == 1, "per-z P(k) slice expected"
        xi_mm = ct.xi.xi_mm_at_r(self.Rfix, k, pk)
        return ct.xi.xi_2halo(self.bias, xi_mm)

    def _pk_to_sigma(self,Rp,k,pk):
        xi_2halo = self._pk_to_xi(k, pk)
        Sigma_mm = ct.deltasigma.Sigma_at_R(Rp, self.Rfix, xi_2halo, self.Md, self.cd, self.omega_m)
        return Sigma_mm, xi_2halo

    def _to_dsigma(self,Rp,sigma):
        dSigma_mm = ct.deltasigma.DeltaSigma_at_R(Rp, Rp, sigma, self.Md, self.cd, self.omega_m)
        return dSigma_mm

    @staticmethod
    def _dsigma_direct(r_grid, sigma_grid, Rp):
        """DeltaSigma = Sigmabar(<R) - Sigma(R) by cumulative trapezoid of
        Sigma R dR on a log-spaced grid extended well below Rp.min(), so
        the two-halo interior mass is integrated rather than modeled."""
        lnr = np.log(r_grid)
        integrand = sigma_grid * r_grid**2  # Sigma R dR = Sigma R^2 dlnR
        cum = np.concatenate([[0.0], np.cumsum(
            0.5*(integrand[1:] + integrand[:-1]) * np.diff(lnr))])
        cum += 0.5 * sigma_grid[0] * r_grid[0]**2  # flat inner disc
        sigma_bar = 2.0 * cum / r_grid**2
        return np.interp(np.log(Rp), lnr, sigma_bar - sigma_grid)

    def pk_to_sigma(self,Rp,k,pk,zvec):
        pk = np.atleast_2d(pk)
        assert pk.shape == (zvec.size, k.size), \
            "P(k, z) must be (n_z, n_k) = (%d, %d), got %r" % (
                zvec.size, k.size, pk.shape)
        self.zvec = zvec
        self.Sigma = np.zeros((zvec.size,Rp.size))
        self.Xi = self.Sigma.copy()
        self._xi_rfix = np.zeros((zvec.size, self.Rfix.size))

        for i in range(zvec.size):
            s, xi_2halo = self._pk_to_sigma(Rp,k,pk[i])
            self.Sigma[i] = s
            self._xi_rfix[i] = xi_2halo
            self.Xi[i] = np.interp(np.log(Rp), np.log(self.Rfix), xi_2halo)
        return

    def pk_to_dsigma(self,Rp,k,pk,zvec=None):
        self.R = Rp
        if zvec is None:
            zvec = np.atleast_1d(0.0)

        # check if Sigma is computed
        if not hasattr(self,'Sigma'):
            self.pk_to_sigma(Rp,k,pk,zvec)

        # convert to delta sigma
        self.dSigma = np.zeros((self.zvec.size,Rp.size))

        if self.dsigma_method == 'direct':
            r_ext = np.logspace(-3., np.log10(Rp.max()), 256)
            for i in range(self.zvec.size):
                sigma_ext = ct.deltasigma.Sigma_at_R(
                    r_ext, self.Rfix, self._xi_rfix[i], self.Md, self.cd, self.omega_m)
                self.dSigma[i] = self._dsigma_direct(r_ext, sigma_ext, Rp)
            return

        # 'sandwich': add an analytic NFW so cluster_toolkit's interior
        # extrapolation in DeltaSigma_at_R is NFW-dominated (hence valid),
        # then subtract the same halo's analytic DeltaSigma. Consistent
        # (Md, cd) throughout makes the dummy halo cancel exactly; the
        # two-halo term's own interior mass below Rp.min() is NOT
        # recovered by this method (use 'direct' for that).
        sig_nfw = sigmaNFW_Analytical(Rp, self.Md, self.cd, rho_c=self.RHO_C*self.omega_m)/1e12
        dsig_nfw = deltaSigmaNFW_Analytical(Rp, self.Md, self.cd, rho_c=self.RHO_C*self.omega_m)/1e12
        for i in range(self.zvec.size):
            self.dSigma[i] = self._to_dsigma(Rp, self.Sigma[i] + sig_nfw) - dsig_nfw


def scaleShiftCosmo(znew, cosmo, eps=1e-9):
    """Scale Shift Cosmology

    To adapt to a new cosmology we can re-scale the distances by taking
    into account the fiducial cosmolgoy.

    scaleShiftCosmo is basically the ratio of the comoving distance
    and the hubble factor
        scaleShiftCosmo = (dist_c/h) / (dist_c_fid/h_fid)

    the fiducial cosmology was set to Omega_m = 0.3 and H0 = 70

    Args:
        znew (array): redshift vector
        cosmo (astropy.cosmology): astropy cosmology

    Returns:
        scale_shift: scale shift factor = ratio of comoving distances/H(z)
        hubble_shift: the hubble flow is the shift on the parallel direction of the los
    """
    import astropy.cosmology
    cosmo_fid = astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    # cosmological quantities
    # h0 = block[cosmo, "h0"]

    # fiducical cosmology
    Hz_fid = cosmo_fid.H(znew).value
    dc_fid = cosmo_fid.comoving_distance(znew).value

    # current cosmology
    Hz = cosmo.H(znew).value
    dc = cosmo.comoving_distance(znew).value

    # scale shift
    scale_shift = dc/(dc_fid+eps)
    scale_shift[0] = 1.

    # hubble shift
    hubble_shift = Hz/Hz_fid

    return scale_shift, hubble_shift


# ===========================================================================
# CosmoSIS module entry points
# ===========================================================================
# Port of y3_buzzard/halo_model_cosmosis.py onto this library (the pipelines
# copy of the same physics), so a pipeline ini can source its [halo_model]
# stage from src/pipelines/cosmology directly. Same options, same datablock
# contract: haloModel/{m_h, lnM, z, rhoc, rho_m_ref,
# one_halo_physical_density, bias}, xi_nl/{r, z, xi_nl}, and the optional
# 1h/2h lensing tables. Entry points live in the owning physics module
# (the src/pipelines/cosmology convention: sigma_crit_inv.py, prj_params.py).

_GSL_HANDLER_DISABLED = False


def _disable_gsl_abort_handler():
    """Swap GSL's abort-on-error handler for the no-op handler.

    cluster_toolkit links libgsl; at extreme cosmologies the qag integrator
    inside cluster_toolkit.peak_height.nu_at_M can fail with a roundoff
    error and the default GSL handler calls abort(), killing the worker.
    With the handler off, GSL returns non-zero status codes and
    cluster_toolkit raises a Python exception the sampler can absorb.
    """
    global _GSL_HANDLER_DISABLED
    if _GSL_HANDLER_DISABLED:
        return
    import ctypes
    try:
        import ctypes.util
        lib_path = ctypes.util.find_library("gsl")
        if lib_path is None:
            # Fallback: rely on cluster_toolkit having loaded libgsl already.
            lib_path = "libgsl.so.25"
        gsl = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        gsl.gsl_set_error_handler_off.restype = ctypes.c_void_p
        gsl.gsl_set_error_handler_off()
        _GSL_HANDLER_DISABLED = True
        print("[cosmology.halo_model] GSL abort-on-error handler disabled "
              f"(via {lib_path})", flush=True)
    except OSError as exc:
        print(f"[cosmology.halo_model] WARNING: could not disable GSL "
              f"abort handler: {exc}", flush=True)


def setup(options):
    from cosmosis.datablock import option_section

    _disable_gsl_abort_handler()

    section = option_section
    # Mpc/h comoving distance, distance on the sky
    R_perp_min = options[section, "R_perp_min"]
    R_perp_max = options[section, "R_perp_max"]
    R_perp_bins = int(options[section, "R_perp_bins"])

    # radii of xi_hm, in Mpc/h
    Radii_min = options[section, "Radii_min"]
    Radii_max = options[section, "Radii_max"]
    Radii_bins = int(options[section, "Radii_bins"])

    # mass (float): Halo mass Msun/h.
    M_min = options[section, "M_min"]
    M_max = options[section, "M_max"]
    M_bins = int(options[section, "M_bins"])

    # Two independent toggles for the lensing stack: 1h publishes
    # Sigma_nfw/dSigma_nfw/concentration (cheap analytic NFW tables); 2h
    # publishes Sigma_hh/dSigma_hh (cluster_toolkit Hankel transforms,
    # the dominant cost). Backward-compat: the old single `compute_lensing`
    # knob maps to both when set; defaults are (1h=T, 2h=T).
    def _bool(key, default):
        try:
            return bool(options.get_bool(section, key, default=default))
        except Exception:
            return default
    compute_lensing = _bool("compute_lensing", True)
    compute_lensing_1h = _bool("compute_lensing_1h", compute_lensing)
    compute_lensing_2h = _bool("compute_lensing_2h", compute_lensing)

    # z_halo: the FIXED redshift at which the 1h concentration is evaluated
    # (owner-ratified convention, issue #3): c(M, z_halo) with z_halo an ini
    # parameter, default 0.4. Density normalisation stays comoving rho_m0
    # regardless (see execute()). `one_halo_z` is a deprecated alias.
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

    # concentration_amplitude: multiply Child18 c(M, z_halo) by this factor
    # (Buzzard clusters are ~1.25x more concentrated than Child18). Applied
    # to BOTH the 1-halo term and the published concentration.
    try:
        c_amp = float(options.get_double(section, "concentration_amplitude",
                                         default=1.0))
    except Exception:
        c_amp = 1.0

    # concentration_fixed: bypass Child18 c(M, z_halo) entirely and use this
    # single mass-independent value for the 1-HALO TERM ONLY (published
    # concentration and Sigma_nfw/dSigma_nfw). Default NaN = disabled
    # (Child18 + concentration_amplitude, as above). shearPrj/Costanzi-2026
    # keeps the concentration-mass relation regardless -- this knob only
    # touches lensModel.c ahead of first_halo_term. Mutually exclusive with
    # concentration_amplitude (ambiguous which one the fixed value already
    # includes).
    try:
        c_fixed = float(options.get_double(section, "concentration_fixed",
                                           default=np.nan))
    except Exception:
        c_fixed = np.nan
    if np.isfinite(c_fixed) and c_amp != 1.0:
        raise ValueError(
            "halo_model: concentration_fixed and concentration_amplitude "
            "are mutually exclusive -- concentration_fixed already IS the "
            "final 1-halo concentration value")

    # one_halo_z_density: redshift at which the 1-halo DENSITY normalisation
    # is evaluated. Default 0.0 = frozen comoving rho_m0. Set >0 for the
    # physical rho_m(z) = rho_m0 (1+z)^3 APPROXIMATION (issue #22): the
    # (1+z)^3 lands outside the C++ z-integral, at the bin centre.
    try:
        z_density = float(options.get_double(section, "one_halo_z_density",
                                             default=0.0))
    except Exception:
        z_density = 0.0

    # one_halo_physical_density: the RIGOROUS in-integrand treatment of the
    # physical mean density via the exact fixed-c NFW identity
    # DSigma_phys(R | z) = (1+z)^2 DSigma_com(R (1+z)); consumers fold the
    # (1+z)^2 into the shear z-weight and rescale the query radius. NUMBER
    # COUNTS ARE NEVER TOUCHED. Requires z_density = 0 (comoving tables).
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

    return (R_perp_min, R_perp_max, R_perp_bins,
            Radii_min, Radii_max, Radii_bins,
            M_min, M_max, M_bins,
            compute_lensing_1h, compute_lensing_2h, one_halo_z, c_amp,
            z_density, physical_density, c_fixed)


def execute(block, config):
    import astropy.cosmology
    from cosmosis.datablock import names

    cosmo_names = names.cosmological_parameters
    section_name = "haloModel"

    (R_perp_min, R_perp_max, R_perp_bins,
     Radii_min, Radii_max, Radii_bins,
     M_min, M_max, M_bins,
     compute_lensing_1h, compute_lensing_2h, one_halo_z, c_amp,
     z_density, physical_density, c_fixed) = config

    # cosmo parameters
    omega_m = block[cosmo_names, "omega_m"]
    omega_b = block[cosmo_names, "omega_b"]
    H0 = block[cosmo_names, "H0"]

    cosmology = astropy.cosmology.FlatLambdaCDM(
        H0*100, omega_m, Ob0=omega_b, Tcmb0=2.725)

    # linear matter P(k) -- used for the bias
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
        # CosmoSIS publishes an un-normalised growth factor (D -> a at
        # early times, so D(0) != 1). bias(M, z) needs the ratio
        # sigma(M, z)/sigma(M, 0) = D(z)/D(0), so renormalise here.
        daz  = np.interp(z, z_az, daz) / np.interp(0.0, z_az, daz)
    else:
        # cp_camb mode without growth_parameters: D(z)=1 keeps the wiring
        # testable but Bias(M, z>0) is wrong -- publish growth_parameters
        # upstream for physics runs.
        daz = np.ones_like(z)

    # compute overdensities; physical not comoving
    rho_c0 = float(cosmology.critical_density0.to('Msun/Mpc^3').value)
    rho_m = omega_m*rho_c0
    rho_mz = rho_m*(1.+z)**3

    # setup bins
    R_perp = np.logspace(np.log10(R_perp_min), np.log10(R_perp_max), R_perp_bins)
    M = np.logspace(np.log10(M_min), np.log10(M_max), M_bins)
    logM = np.log(M)

    # Bias(M, z): peak height at z=0 from cluster_toolkit, evolved with the
    # growth function D(z). Shape (z.size, M.size).
    bM = biasModel(k_h, P_k[0], omega_m, odelta=200)
    nu = bM.nu_at_M(M)
    Bias = np.array([bM.bias_at_nu(nu/dazi) for dazi in daz])

    # xi_NL(r, z) pre-tabulated on a fixed log-r grid for the C++
    # b_sel_marg / sigma_prj integrands; always needed by the
    # Costanzi-2026 projection pipeline.
    r_xi = np.logspace(-3.0, 3.0, 128)  # Mpc/h comoving
    xi_NL = np.zeros((z.size, r_xi.size))
    for iz, zz in enumerate(z):
        xi_NL[iz] = ct.xi.xi_mm_at_r(r_xi, k_nl, P_k_nl[iz])

    # ----- always-on datablock writes (consumed by bsel + shear_prj) -----
    block[section_name, "m_h"] = M
    block[section_name, "lnM"] = logM
    block[section_name, "z"] = z
    block[section_name, "rhoc"] = rho_mz
    # rho_m_ref: THE single NFW reference density every profile consumer
    # uses for BOTH the halo boundary r_200 = [3M/(800 pi rho_m_ref)]^(1/3)
    # and the amplitude rho_s = delta_c * rho_m_ref (unified-convention
    # decision 2026-08-24). h-unit convention: masses Msun/h, radii Mpc/h,
    # so rho_ref is the h-unit critical density -- the SAME constant
    # lensingModel.rhoc0 uses for the centred tables.
    RHOC_HUNITS = 2.77533742639e+11
    block[section_name, "rho_m_ref"] = (omega_m * RHOC_HUNITS
                                        * (1.0 + z_density) ** 3)
    # 0/1 flag for the rigorous physical-density treatment (see setup).
    block[section_name, "one_halo_physical_density"] = int(physical_density)
    block[section_name, "bias"] = Bias
    # xi_NL(r, z) table for the b_sel_marg / sigma_prj integrands
    block["xi_nl", "r"]     = r_xi
    block["xi_nl", "z"]     = z
    block["xi_nl", "xi_nl"] = xi_NL

    # ----- lensing section: optional -------------------------------------
    if compute_lensing_1h or compute_lensing_2h:
        lensModel = lensingModel(R_perp, omega_m, 200)
        scaleShift, hubbleShift = scaleShiftCosmo(z, cosmology)
        block[section_name, "r_sigma"] = R_perp
        block[section_name, "scale_shift"] = scaleShift
        block[section_name, "hubble_shift"] = hubbleShift
        block[section_name, "k"] = k_h

    if compute_lensing_1h:
        if np.isfinite(c_fixed):
            # concentration_fixed: mass-independent 1-halo concentration,
            # bypassing Child18 entirely. Pre-setting self.c makes
            # first_halo_term skip its own concentration_at_M call.
            lensModel.c = c_fixed * np.ones_like(M)
        else:
            # one_halo_z enters ONLY the Child18 concentration. The density
            # normalisation stays the COMOVING rho_m0 (z_density=0 default);
            # setting self.c first makes first_halo_term skip its own z=0
            # recompute.
            lensModel.concentration_at_M(M, z=one_halo_z,
                                         model_name="Child18")
            if c_amp != 1.0:
                lensModel.c = lensModel.c * c_amp
        lensModel.first_halo_term(M, z=z_density, conc_model_name="Child18")
        block[section_name, "Sigma_nfw"]  = lensModel.Sigma['1h']
        block[section_name, "dSigma_nfw"] = lensModel.dSigma['1h']
        block[section_name, "concentration"] = lensModel.c

    if compute_lensing_2h:
        lensModel.second_halo_term(z, k_nl, P_k_nl)
        block[section_name, "Sigma_hh"]  = lensModel.Sigma['2h']
        block[section_name, "dSigma_hh"] = lensModel.dSigma['2h']

    return 0


def cleanup(config):
    return 0
