"""Build Sigma_critical^-1(z_lens, R) from a beta lookup table.

Port of ``y3_buzzard/buildSigmaCritInv.py`` (the ``sigmaCritInv``
producer the reviewer asked for in ``src/pipelines/cosmology``). The
``y3_buzzard`` original stays in place — this is a copy into the
canonical shared location, not a move; see ``README.md`` in this
directory for why. The only change from the original is dropping the
unused ``from setup_bins import zmeans`` import.

Computes sigma critical as a function of the lens redshift using a
lookup table of the effective source redshifts.

In the weak lensing regime the sigma_critical is a statistical
measurement of the ensemble of sources.

The effective sigma_critical can be defined as:
    <sigma_critical(zL, zs_eff)> = 4piG/c^2 da_l <beta(zl, zs_eff)>
Where beta is the ratio beta = dls/ds.
For a redshift probability distribution the mean beta
is the average of the redshift sources.
    <beta(zs_eff)> = \\int p(zs) d(zl,zs)/d(zs) dzs
So the effective source redshift is:
    d(zl,zs_eff)/d(zs_eff) = <beta(zl, zs)>
which is a single number for a given lens redshift.

We import beta from a lookup table. Since the quantity is cosmological
we need to apply a cosmological shift on the distance. This correction
is:
    beta = 1 - scale_ratio(zs, zl) * (1-beta_fid)

where beta_fid is the one computed by the fiducial cosmology and
    scale_ratio(zs, zl) = scaleShiftCosmo(zs)/scaleShiftCosmo(zl)

the scaleShiftCosmo is basically the ratio of the comoving distance
and the hubble factor
    scaleShiftCosmo = (dist_c/h) / (dist_c_fid/h_fid)

for each cosmological step we compute the cosmological shift
on the ``halo_model.scaleShiftCosmo`` function.

Author: Johnny Esteves
Created: 17th Nov, 2022
"""
import os
import numpy as np
from scipy.interpolate import interp1d
from cosmosis.datablock import option_section, names

cosmo = names.cosmological_parameters

# for test purpose only
fake_data = False

def setup(options):
        section = option_section

        # loads beta table
        # beta_lut = options[section, "beta_lut"]
        # beta_lut_fname = os.path.expandvars(str(beta_lut))
        beta_lut_fname = os.path.expandvars('${Y3_CLUSTER_CPP_DIR}/data/beta_sigCrit_lut_mock_buzzard_zl.npz')
        betaTable = np.load(beta_lut_fname)

        # betaTable.keys() -> "zsrc", "rbins", "beta"
        # beta has ndshape (zbins.size, rbins.size)

        # interpolates beta over lens redshift
        z_beta = betaTable['zlens']
        r_beta = betaTable['Radii']
        z_src = betaTable['zEff'] # ndshape (zlens.size, rbins.size)
        beta_fid = betaTable["betaEff"] # ndshape (zbins.size, rbins.size)

        is_delta_sigma = bool(options[section, "is_delta_sigma"])

        # fake data
        if fake_data:
            z_beta = np.linspace(0.1, 1.2, 64)
            r_beta = np.logspace(-2, 2., 32)

            beta_fid = np.zeros((z_beta.size, r_beta.size))
            z_src = np.zeros((z_beta.size, r_beta.size))
            z_src = z_src + (1+0.2)*(z_beta[:,np.newaxis])

        return z_beta, r_beta, z_src, beta_fid, is_delta_sigma

def execute(block, config):
        z_beta, r_beta, z_src, beta_fid, is_delta_sigma  = config

        # setup constants
        # 4piG/c^2
        const = 6.01e-19 # Mpc/M_sol
        const*= 1e6 # pc/M_sol

        # cosmological quantities
        z_da = block["distances",'z']
        da = block["distances",'d_a'] # Mpc

        # lenses redshift is equal to the cluster mean redshifts
        section_name = "haloModel"
        z_lenses = block["haloModel", "z"]
        r_sigma = block[section_name, "r_sigma"]
        shift = block[section_name, "scale_shift"]
        z_shift = z_lenses

        # lens comoving distance
        dc_lenses = interp1d(z_da, da*(1+z_da))(z_lenses)

        # cosmological shift := (H/Hfid)*(dc/dc_fid)
        # fid stands for the fiducial cosmology
        # compute the shift ratio of dls/ds
        scale_shift_func = interp1d(z_shift, shift, bounds_error=False)

        # compute beta
        zsize = z_lenses.size
        sigma_crit_inv = np.zeros((zsize, r_sigma.size))
        # beta with the cosmological correction
        for i in range(zsize):
                zL = z_lenses[i]
                # find zl indice in z_beta_lenses vector
                ix = np.interp(zL, z_beta, range(z_beta.size)).astype(int)
                zS = z_src[ix] # effective source redshift vector with r_beta size
                betaFid = beta_fid[ix] # vector with r_beta size

                # compute scale factor
                scale_src = scale_shift_func(zS)
                scale_lens= scale_shift_func(zL)
                scale_ratio = scale_src/scale_lens

                # beta is unitless
                beta_zl = 1 - scale_ratio*(1-betaFid)
                # if scale_ratio is one, beta equals to beta_fid

                # compute sigma_crit_inv [pc^2/Msun]
                sigma_crit_inv_vec = const * (dc_lenses[i]*1e6) * beta_zl

                # for test purpose only
                # sigma_crit_inv_vec =  const * (dc_lenses[i]*1e6) * np.ones_like(r_beta)

                # interpolate with the new binning scheme
                # sigma_crit_inv[i] = interp1d(r_beta, sigma_crit_inv_vec, bounds_error=False)(r_sigma)
                sigma_crit_inv[i] = np.interp(r_sigma, r_beta, sigma_crit_inv_vec)

        # JTA hack hack hack
        if is_delta_sigma:
                sigma_crit_inv = sigma_crit_inv/sigma_crit_inv
                # set sigma_crit_inv to unity
                # then, shear is actually delta sigma

        block["sigmaCritInv", "r_sigma"] = r_sigma
        block["sigmaCritInv", "z"] = z_lenses
        block["sigmaCritInv", "sigma_crit_inv"] = sigma_crit_inv
        return 0

def cleanup(config):
	pass
