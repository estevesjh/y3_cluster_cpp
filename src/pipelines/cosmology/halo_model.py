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
import numpy as np
import cluster_toolkit as ct

from .nfw_model import sigmaNFW_Analytical, deltaSigmaNFW_Analytical
from .concentration import child18_mass_concentration, duffy_concentration_relation


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

    def __init__(self,omega_m=0.3,exclusion=True,NSIZE=50, Md=1e13, cd=4.,bias=1.00):
        self.omega_m = omega_m
        self.NSIZE = NSIZE
        self.Rfix = np.logspace(-3., 3., NSIZE, base=10) #Xi_hm MUST be evaluated to higher than BAO
        self.Md, self.cd = Md, cd # dummy values
        self.bias = bias
        self.exclusion = exclusion

    def _pk_to_sigma(self,Rp,k,pk):
        ## TODO: add NFW profile
        xi_mm    = ct.xi.xi_mm_at_r(self.Rfix, k, pk)
        xi_2halo = ct.xi.xi_2halo(self.bias, xi_mm)
        Sigma_mm = ct.deltasigma.Sigma_at_R(Rp, self.Rfix, xi_2halo, self.Md, self.cd, self.omega_m)
        return Sigma_mm, np.interp(np.log(Rp), np.log(self.Rfix), xi_2halo)

    def _to_dsigma(self,Rp,sigma):
        dSigma_mm = ct.deltasigma.DeltaSigma_at_R(Rp, Rp, sigma, self.Md/10., self.cd, self.omega_m)
        return dSigma_mm

    def pk_to_sigma(self,Rp,k,pk,zvec):
        self.zvec = zvec
        self.Sigma = np.zeros((zvec.size,Rp.size))
        self.Xi = self.Sigma.copy()

        for i in range(zvec.size):
            s, xi = self._pk_to_sigma(Rp,k,pk)
            self.Sigma[i] = s
            self.Xi[i] = xi
        return

    def pk_to_dsigma(self,Rp,k,pk,zvec=None):
        self.R = Rp

        if zvec is None:
            self.dSigma = np.zeros((Rp.size,))
            sigma = self._pk_to_sigma(Rp, k, pk)
            if self.exclusion:
                ## ADD NFW
                sigma+= sigmaNFW_Analytical(Rp, self.Md, self.cd, rho_c=self.RHO_C*self.omega_m)/1e12

            self.dSigma = self._to_dsigma(Rp, sigma)
            # if self.exclusion:
                ## Subtract NFW
                # self.dSigma -= 0.993*deltaSigmaNFW_Analytical(Rp, self.Md, self.cd, rho_c=self.RHO_C*self.omega_m)/1e12
            self.dSigma -= deltaSigmaNFW_Analytical(Rp, self.Md, self.cd, rho_c=self.RHO_C*self.omega_m)/1e12

            self.dSigma = np.where(self.dSigma<0.,np.nan,self.dSigma)
            return


        # check if Sigma is computed
        if not hasattr(self,'Sigma'):
            self.pk_to_sigma(Rp,k,pk,zvec)

        # convert to delta sigma
        self.dSigma = np.zeros((self.zvec.size,Rp.size))
        for i in range(self.zvec.size):
            sigma = self.Sigma[i]
            if self.exclusion:
                ## ADD NFW
                sigma+= sigmaNFW_Analytical(Rp, self.Md, self.cd, rho_c=self.RHO_C*self.omega_m)/1e12

            self.dSigma[i] = self._to_dsigma(Rp, sigma)
            if self.exclusion:
                ## Subtract NFW
                self.dSigma[i] -= deltaSigmaNFW_Analytical(Rp, self.Md, self.cd, rho_c=self.RHO_C*self.omega_m)/1e12

            self.dSigma = np.where(self.dSigma<0.,np.nan,self.dSigma)

        pass


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
