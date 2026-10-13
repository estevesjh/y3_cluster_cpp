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
        self.sigma_bar = {'1h':None, '2h':None}
        self.m3d = {'1h':None, '2h':None}
    
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
        self.sigma_bar['1h'] = self.Sigma['1h'] + self.dSigma['1h']
    
    def second_halo_term(self, z, k, P):
        ## Cluster Toolkit Halo-Halo projected \Sigma and \Delta \Sigma second halo term computation
        ## It also computed the 2-point correlation function \Xi_{mm}
        # The mass and concentration are dummy values
        # The redshift is needed if the power-spectrum is non linear
        p2h = ct_2hTerm(self.omega_m, Md=1e14, cd=5, bias=1.0)
        p2h.pk_to_dsigma(self.R, k, P, z)
        self.Sigma['2h'] = p2h.Sigma
        self.dSigma['2h'] = p2h.dSigma
        self.sigma_bar['2h'] = p2h.sigma_bar
        self.Wp = p2h.Xi
        rho_m_h = self.omega_m * p2h.RHO_C
        self.m3d['2h'] = ct_2hTerm.m3d_at_r(self.R, self.Wp, rho_m=rho_m_h)

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


class max1h2hOperators(object):
    """Operators for 3D halo-exclusion max(1h, 2h) profile assembly.

    Computes the 2D matching tables (shape: (n_z, n_M)) that bridge the
    1-halo core and the decoupled 2-halo outer tail:
        - rtrans: 3D crossing scale where xi_1h = b * xi_nl
        - c0: Decoupled 3D spherical core mass difference / pi
        - delta_gap_0: Surface density boundary gap at rtrans (Delta_gap^0)
        - delta_gap_1: Excess surface density boundary gap at rtrans (Delta_gap^1)
    """

    def __init__(self, lens_model):
        self.lens_model = lens_model

    @staticmethod
    def _interp_2d_log_r(ln_rq, ln_r_grid, data_2d):
        """Piecewise-linear interpolation along log(r) for 2D arrays shape (n_z, n_M)."""
        d = np.diff(ln_r_grid)
        dln_r = d[0]
        rows = np.arange(ln_rq.shape[0])[:, None]
        # Fast path: uniform log-spaced grid (standard in cosmology pipelines)
        if np.all(np.abs(d - dln_r) < 1e-6 * np.abs(dln_r)):
            fj = (ln_rq - ln_r_grid[0]) / dln_r
            j = np.clip(fj.astype(int), 0, len(ln_r_grid) - 2)
            w = fj - j
            y0 = data_2d[rows, j]
            y1 = data_2d[rows, j + 1]
            out = y0 + w * (y1 - y0)
        else:
            j = np.clip(np.searchsorted(ln_r_grid, ln_rq) - 1, 0, len(ln_r_grid) - 2)
            x0 = ln_r_grid[j]
            x1 = ln_r_grid[j + 1]
            y0 = data_2d[rows, j]
            y1 = data_2d[rows, j + 1]
            slope = (y1 - y0) / (x1 - x0)
            out = y0 + slope * (ln_rq - x0)
        out = np.where(ln_rq < ln_r_grid[0], data_2d[rows, 0], out)
        out = np.where(ln_rq > ln_r_grid[-1], data_2d[rows, -1], out)
        return out

    def compute_rtrans(self, M, z, bias, r_xi, xi_nl, conc=None):
        """Compute the 2D crossing scale r_trans(M, z) where xi_1h = b * xi_nl.

        Parameters
        ----------
        M : array_like, shape (n_M,)
            Halo masses (Msun/h).
        z : array_like, shape (n_z,)
            Redshifts.
        bias : array_like, shape (n_z, n_M)
            Halo bias b(M, z).
        r_xi : array_like, shape (n_r,)
            Comoving radial grid (Mpc/h).
        xi_nl : array_like, shape (n_z, n_r)
            Nonlinear matter correlation function xi_nl(r, z).
        conc : array_like, shape (n_M,), optional
            Halo concentrations. If None, uses self.lens_model.c.

        Returns
        -------
        r_trans : ndarray, shape (n_z, n_M)
            Crossing radius in Mpc/h.
        """
        M_arr = np.atleast_1d(M)
        bias_arr = np.atleast_2d(bias)
        r_arr = np.asarray(r_xi, dtype=float)
        xi_arr = np.atleast_2d(xi_nl)

        if conc is None:
            conc = getattr(self.lens_model, 'c', None)
            if conc is None:
                conc = child18_mass_concentration(M_arr, 0.0, halo_sample='stacked_nfw')
        c_arr = np.asarray(conc, dtype=float)
        if c_arr.ndim == 0 or c_arr.size == 1:
            c_arr = np.full(M_arr.size, float(c_arr))

        rho_m = self.lens_model.rhom0
        r200 = (3.0 * M_arr / (800.0 * np.pi * rho_m))**(1.0 / 3.0)
        rs = r200 / c_arr
        delta_c = (200.0 / 3.0) * c_arr**3 / (np.log(1.0 + c_arr) - c_arr / (1.0 + c_arr))
        rho_s = delta_c * rho_m

        # Window optimization: crossing r_trans lies strictly in [0.2, 50.0] Mpc/h
        # across all cluster masses (10^12 to 10^16 Msun/h) and redshifts (0 to 3)
        mask = (r_arr >= 0.2) & (r_arr <= 50.0)
        if np.any(mask):
            idx_win = np.where(mask)[0]
            i0 = max(0, idx_win[0] - 1)
            i1 = min(len(r_arr), idx_win[-1] + 2)
        else:
            i0, i1 = 0, len(r_arr)
        r_sub = r_arr[i0:i1]

        # xi_1h on subgrid has shape (1, n_M, n_sub)
        x_sub = r_sub[None, :] / rs[:, None]
        xi_1h_sub = rho_s[:, None] / (rho_m * x_sub * (1.0 + x_sub)**2) - 1.0

        # xi_2h on subgrid has shape (n_z, n_M, n_sub)
        diff_sub = xi_1h_sub[None, :, :] - (bias_arr[:, :, None] * xi_arr[:, None, i0:i1])

        # Bracket crossing detection on subgrid
        is_crossing = (diff_sub[:, :, :-1] >= 0) & (diff_sub[:, :, 1:] <= 0)
        idx_cross = np.argmax(is_crossing, axis=-1) + i0
        ra = r_arr[idx_cross]
        rb = r_arr[idx_cross + 1]

        has_cross = np.any(is_crossing, axis=-1)
        ra = np.where(has_cross, ra, np.maximum(r_arr[0], 0.25 * r200[None, :]))
        rb = np.where(has_cross, rb, np.minimum(r_arr[-1], 4.0 * r200[None, :]))

        # Vectorized bisection over (n_z, n_M) with O(1) grid interpolation
        ln_r_arr = np.log(r_arr)
        for _ in range(5):
            rmid = np.sqrt(ra * rb)
            xmid = rmid / rs[None, :]
            x1h_mid = rho_s[None, :] / (rho_m * xmid * (1.0 + xmid)**2) - 1.0
            x2h_mid = bias_arr * self._interp_2d_log_r(np.log(rmid), ln_r_arr, xi_arr)
            f_mid = x1h_mid - x2h_mid
            ra = np.where(f_mid > 0, rmid, ra)
            rb = np.where(f_mid <= 0, rmid, rb)

        return np.sqrt(ra * rb)

    def compute_c0(self, r_trans, M, z, bias, r_xi, m3d_hh, conc=None):
        """Compute the 2D core mass decoupling constant C_0(M, z) in (Msun/h).

        C_0(M, z) = [M_3D,1h(<r_trans) - b(M, z) * M_3D,2h(<r_trans)] / pi
        """
        M_arr = np.atleast_1d(M)
        bias_arr = np.atleast_2d(bias)
        r_arr = np.asarray(r_xi, dtype=float)
        m3d_2h_arr = np.atleast_2d(m3d_hh)

        if conc is None:
            conc = getattr(self.lens_model, 'c', None)
            if conc is None:
                conc = child18_mass_concentration(M_arr, 0.0, halo_sample='stacked_nfw')
        c_arr = np.asarray(conc, dtype=float)
        if c_arr.ndim == 0 or c_arr.size == 1:
            c_arr = np.full(M_arr.size, float(c_arr))

        rho_m = self.lens_model.rhom0
        r200 = (3.0 * M_arr / (800.0 * np.pi * rho_m))**(1.0 / 3.0)
        rs = r200 / c_arr
        delta_c = (200.0 / 3.0) * c_arr**3 / (np.log(1.0 + c_arr) - c_arr / (1.0 + c_arr))
        rho_s = delta_c * rho_m

        ln_r_arr = np.log(r_arr)
        xk = r_trans / rs[None, :]
        m1h = (4.0 * np.pi * rho_s[None, :] * (rs[None, :]**3) * (np.log(1.0 + xk) - xk / (1.0 + xk))
               - (4.0 / 3.0) * np.pi * rho_m * r_trans**3)
        m2h = self._interp_2d_log_r(np.log(r_trans), ln_r_arr, m3d_2h_arr)

        return (m1h - bias_arr * m2h) / np.pi

    def compute_delta_gap_0(self, r_trans, M, z, bias, conc=None, z_density=0.0):
        """Compute the 2D surface density boundary gap Delta_gap^0(M, z) in (Msun/pc^2).

        Delta_gap^0(M, z) = b(M, z) * Sigma_hh(r_trans, z) - Sigma_1h(r_trans, M)
        """
        M_arr = np.atleast_1d(M)
        bias_arr = np.atleast_2d(bias)

        if conc is None:
            conc = getattr(self.lens_model, 'c', None)
            if conc is None:
                conc = child18_mass_concentration(M_arr, 0.0, halo_sample='stacked_nfw')
        c_arr = np.asarray(conc, dtype=float)
        if c_arr.ndim == 0 or c_arr.size == 1:
            c_arr = np.full(M_arr.size, float(c_arr))

        rho_c_z = self.lens_model.rhom0 * (1.0 + z_density)**3
        ln_R_2h = np.log(self.lens_model.R)
        sig_2h = np.atleast_2d(self.lens_model.Sigma['2h'])

        s2h = self._interp_2d_log_r(np.log(r_trans), ln_R_2h, sig_2h)
        s1h = sigmaNFW_Analytical(r_trans, M_arr[None, :], c_arr[None, :], rho_c=rho_c_z) / 1e12

        return bias_arr * s2h - s1h

    def compute_delta_gap_1(self, r_trans, c0, M, z, bias, conc=None, z_density=0.0):
        """Compute the 2D excess surface density boundary gap Delta_gap^1(M, z) in (Msun/pc^2).

        Delta_gap^1(M, z) = [b(M, z) * DeltaSigma_hh(r_trans, z) + c0 / r_trans^2] - DeltaSigma_1h(r_trans, M)
        """
        M_arr = np.atleast_1d(M)
        bias_arr = np.atleast_2d(bias)

        if conc is None:
            conc = getattr(self.lens_model, 'c', None)
            if conc is None:
                conc = child18_mass_concentration(M_arr, 0.0, halo_sample='stacked_nfw')
        c_arr = np.asarray(conc, dtype=float)
        if c_arr.ndim == 0 or c_arr.size == 1:
            c_arr = np.full(M_arr.size, float(c_arr))

        rho_c_z = self.lens_model.rhom0 * (1.0 + z_density)**3
        ln_R_2h = np.log(self.lens_model.R)
        dsig_2h = np.atleast_2d(self.lens_model.dSigma['2h'])

        ds2h = self._interp_2d_log_r(np.log(r_trans), ln_R_2h, dsig_2h)
        ds1h = deltaSigmaNFW_Analytical(r_trans, M_arr[None, :], c_arr[None, :], rho_c=rho_c_z) / 1e12
        c0_term = (c0 / (r_trans * r_trans)) / 1e12

        return (bias_arr * ds2h + c0_term) - ds1h


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
        self.Sigma = None
        self.dSigma = None
        self.sigma_bar = None
        self.m3d = None

    @classmethod
    def m3d_at_r(cls, r_grid, xi_grid, rho_m=None, n_gl=5):
        r"""Enclosed 3D mass profile M_3D(<r):
            M_3D(<r) = 4*pi * rho_m * \int_0^r r'^2 \xi(r') dr'
                     = 4*pi * rho_m * \int_0^r r'^3 \xi(r') d\ln r'
        evaluated via Gauss-Legendre quadrature on logarithmic intervals.
        Below r_grid[0], assumes an analytic flat core:
            M_3D(<r_0) = (4*pi/3) * rho_m * xi(r_0) * r_0^3.

        Parameters
        ----------
        r_grid : array_like, shape (n_r,)
            Comoving radial grid (Mpc/h), must be strictly increasing.
        xi_grid : array_like, shape (n_r,) or (n_z, n_r)
            Correlation function values on r_grid.
        rho_m : float or array_like, optional
            Background matter density (Msun * h^2 / Mpc^3). If None,
            uses 1.0 (returning the volume integral / mass per unit density).
        n_gl : int, optional
            Gauss-Legendre quadrature order per log interval (default: 5).

        Returns
        -------
        m3d : ndarray, shape (n_r,) or (n_z, n_r)
            Enclosed 3D mass profile M_3D(<r) in Msun/h (or (Mpc/h)^3 if rho_m=1).
        """
        r_arr = np.asarray(r_grid, dtype=float)
        xi_arr = np.asarray(xi_grid, dtype=float)
        is_1d = (xi_arr.ndim == 1)
        if is_1d:
            xi_arr = xi_arr[np.newaxis, :]

        lnr = np.log(r_arr)
        dlnr = np.diff(lnr)
        xg, wg = np.polynomial.legendre.leggauss(n_gl)
        t_nodes = lnr[:-1, None] + 0.5 * dlnr[:, None] * (xg[None, :] + 1.0)
        w_nodes = 0.5 * dlnr[:, None] * wg[None, :]
        exp_3t = np.exp(3.0 * t_nodes)

        # Gauss-Legendre quadrature on logarithmic intervals. Within interval i
        # [lnr[i], lnr[i+1]], quadrature nodes are t = lnr[i] + u * dlnr[i]
        # where u = 0.5 * (xg + 1). Direct linear interpolation between knots:
        #   xi(t) = xi_i + u * (xi_{i+1} - xi_i)
        # evaluates in pure NumPy vectorization with zero spline overhead or
        # knot search.
        u = 0.5 * (xg + 1.0)
        xi_left = xi_arr[:, :-1, np.newaxis]
        xi_right = xi_arr[:, 1:, np.newaxis]
        xi_eval = xi_left + (xi_right - xi_left) * u[np.newaxis, np.newaxis, :]
        integrand = exp_3t[None, :, :] * xi_eval
        steps = np.sum(w_nodes[None, :, :] * integrand, axis=-1)

        cum = np.cumsum(steps, axis=-1)
        zeros = np.zeros((xi_arr.shape[0], 1), dtype=float)
        cum = np.concatenate([zeros, cum], axis=-1)
        cum += (1.0 / 3.0) * (r_arr[0]**3) * xi_arr[:, 0:1]
        cum *= 4.0 * np.pi

        if rho_m is not None:
            cum = cum * rho_m

        return cum[0] if is_1d else cum

    # Physical aliases and convenience methods
    mass_3d_at_r = m3d_at_r
    enclosed_mass_3d = m3d_at_r
    m3d = m3d_at_r
    int_r2_xi = m3d_at_r
    int_r2_xi_nl = m3d_at_r
    cumulative_volume_integral = m3d_at_r

    def mass_3d(self, r_grid=None, rho_m=None, n_gl=5):
        """Enclosed 3D mass profile M_3D,hh(<r) for the 2-halo term (b=1) via GL."""
        if r_grid is None:
            r_grid = self.Rfix
        if rho_m is None:
            rho_m = self.omega_m * self.RHO_C
        if not hasattr(self, '_xi_rfix') or self._xi_rfix is None:
            raise RuntimeError("pk_to_sigma must be called before mass_3d")
        return self.m3d_at_r(r_grid, self._xi_rfix, rho_m=rho_m, n_gl=n_gl)

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
        if not hasattr(self,'Sigma') or self.Sigma is None:
            self.pk_to_sigma(Rp,k,pk,zvec)

        # convert to delta sigma
        self.dSigma = np.zeros((self.zvec.size,Rp.size))

        if self.dsigma_method == 'direct':
            r_ext = np.logspace(-3., np.log10(Rp.max()), 256)
            for i in range(self.zvec.size):
                sigma_ext = ct.deltasigma.Sigma_at_R(
                    r_ext, self.Rfix, self._xi_rfix[i], self.Md, self.cd, self.omega_m)
                self.dSigma[i] = self._dsigma_direct(r_ext, sigma_ext, Rp)
        else:
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

        self.sigma_bar = self.Sigma + self.dSigma
        return

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
