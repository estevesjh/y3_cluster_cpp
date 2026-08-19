import numpy as np
import cluster_toolkit as ct
from nfwModel import sigmaNFW_Analytical, deltaSigmaNFW_Analytical

#############################################
# Author: Johnny Esteves
# Created: May 16, 2024
#############################################

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

def child18_mass_concentration(M200c, z, halo_sample = 'stacked_nfw'):
	"""
	The concentration mass relation model of Child et al 2018.

    We assume that this is an universal relation that does not depend on cosmology

	Parameters
	-----------------------------------------------------------------------------------------------
	M200c: array_like
		Halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	halo_sample: str
		Can be ``individual_all`` (default), ``individual_relaxed`` (the mean concentration of 
		individual, relaxed halos), ``stacked_nfw`` (the stacked profile with with an NFW profile), 
		and ``stacked_einasto`` (the stacked profile with with an Einasto profile).
		
	Returns
	-----------------------------------------------------------------------------------------------
	c: array_like
		Halo concentration; has the same dimensions as ``M``.
	"""

	if halo_sample == 'individual_all':
		m = -0.10
		A = 3.44
		b = 430.49
		c0 = 3.19
	elif halo_sample == 'individual_relaxed':
		m = -0.09
		A = 2.88
		b = 1644.53
		c0 = 3.54
	elif halo_sample == 'stacked_nfw':
		m = -0.07
		A = 4.61
		b = 638.65
		c0 = 3.59
	elif halo_sample == 'stacked_einasto':
		m = -0.01
		A = 63.2
		b = 431.48
		c0 = 3.36
	else:
		raise Exception('Unknown halo sample for child18 concentration model, %s.' % (halo_sample))

	Mstar = peakHeight_nonLinearMass(z)
	M_MT = M200c / (Mstar * b)	
	c200c = c0 + A * (M_MT**m * (1.0 + M_MT)**-m - 1.0)
	return c200c

def peakHeight_nonLinearMass(z):
    """peakHeight_nonLinearMass 
    The non-linear mass. E.g. eqaution 13 in Child et al. 2018
    Mstar is log-linear with redshift. 
    log(Mstar) = 12.5 - 1.5*z

    You can check with Colossus
    How to generate the data:
    from colossus.cosmology import cosmology
    from colossus.lss.peaks import nonLinearMass
    cosmology.setCosmology('WMAP7-only')
    z = np.linspace(0, 2, 100)
    Mstar = nonLinearMass(z)
    """
    return 10**(12.5-1.5*z)

def duffy_concentration_relation(m_h, z_eff=0.4):
    a_eff = 1/(1+z_eff)
    m_h_pivot = 2e12
    return 7.85*np.power(m_h/m_h_pivot,-0.081)*np.power(a_eff,0.71)

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
