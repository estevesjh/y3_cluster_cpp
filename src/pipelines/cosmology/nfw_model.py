"""Analytical Wright & Brainerd (2000) NFW Sigma/DeltaSigma profile.

Verbatim port of ``y3_buzzard/nfwModel.py`` (Eqns 11 and 14, Wright &
Brainerd 2000: https://iopscience.iop.org/article/10.1086/308744). The
``y3_buzzard`` original stays in place — this is a copy into the
canonical shared location, not a move; see ``README.md`` in this
directory for why.
"""
import numpy as np
from scipy.interpolate import interp1d


def sigmaNFW_Analytical(radii, m200, c, rho_c=0.0):
    """Analytical NFW Surface Mass Density (Eqn 11, Wright & Brainerd 2000).

    Args:
        radii (array): projected radii
        m200 (array): M200 critical mass in solar masses unit
        c (array): concentration
        rho_c (float): critical density at the halo's redshift
    """
    r_virial = convert_m200_to_r200(m200, rho_c)
    rs = r_virial / c

    # eqn 3
    deltac = (200. / 3.) * c**3 / (np.log(1 + c) - c / (1 + c))

    # eqn 15 an 16
    # interpolation avoid numerical issues
    xvec = np.logspace(-3., 4., 1000)
    xvec = np.append(0, xvec)
    fx = interp1d(xvec, f_nfw(xvec))(radii / rs)
    return 2 * rs * deltac * rho_c * fx


def deltaSigmaNFW_Analytical(radii, m200, c, rho_c=0.0):
    """Analytical NFW Delta Sigma (Eqn 14, Wright & Brainerd 2000).

    Args:
        radii (array): projected radii
        m200 (array): M200 critical mass in solar masses unit
        c (array): concentration
        rho_c (float): critical density at the halo's redshift
    """
    r_virial = convert_m200_to_r200(m200, rho_c)
    rs = r_virial / c

    # eqn 3
    deltac = (200. / 3.) * c**3 / (np.log(1 + c) - c / (1 + c))

    # eqn 15 an 16
    # interpolation avoid numerical issues
    xvec = np.logspace(-3., 4., 1000)
    xvec = np.append(0, xvec)
    gx = interp1d(xvec, g_nfw(xvec))(radii / rs)
    return rs * deltac * rho_c * gx


### Compute \Sigma Helper Functions
def f_greater_than(x):
    res = 1. / (x**2 - 1.0)
    res *= (1 - 2.0 / np.sqrt(x**2 - 1.0) * np.arctan(np.sqrt((x - 1.0) / (1.0 + x))))
    return res


def f_less_than(x):
    res = 1. / (x**2 - 1.0)
    res *= (1 - 2.0 / np.sqrt(1.0 - x**2) * np.arctanh(np.sqrt((1.0 - x) / (1.0 + x))))
    return res


def f_nfw(x, eps=1e-9):
    """Analytical normalized Sigma profile, Wright & Brainerd 2000.

    Args:
        x (array): Rp/Rs where Rs is the scale radius, Rs = R200/c
    """
    if (isinstance(x, float)) or (isinstance(x, int)):
        x = np.array([x])

    res = 1 / 3. * np.ones_like(x)

    ix = np.where(x <= 1 - eps)[0]
    res[ix] = f_less_than(x[ix])

    ix = np.where(x >= 1 + eps)[0]
    res[ix] = f_greater_than(x[ix])
    return res


### Compute \Delta \Sigma Helper Functions
def g_less_than(x, core=4e-3):
    # below core the solution fails numereically
    x = np.where(x < core, core, x)
    term1 = 8.0 * np.arctanh(np.sqrt((1.0 - x) / (1.0 + x))) / (x**2 * np.sqrt(1.0 - x**2))
    term2 = 4.0 / x**2 * np.log(x / 2.0)
    term3 = -2.0 / (x**2 - 1.0)
    term4 = 4.0 * np.arctanh(np.sqrt((1.0 - x) / (1.0 + x))) / ((x**2 - 1.0) * np.sqrt(1.0 - x**2))
    return term1 + term2 + term3 + term4


def g_greater_than(x, xmax=1e9):
    # above xmax the function achieves machine precision
    x = np.where(x > xmax, xmax, x)
    term1 = 8.0 * np.arctan(np.sqrt((x - 1.0) / (1.0 + x))) / (x**2 * np.sqrt(x**2 - 1.0))
    term2 = 4.0 / x**2 * np.log(x / 2.0)
    term3 = -2.0 / (x**2 - 1.0)
    term4 = 4.0 * np.arctan(np.sqrt((x - 1.0) / (1.0 + x))) / ((x**2 - 1.0)**(3.0 / 2.0))
    return term1 + term2 + term3 + term4


def g_nfw(x, eps=1e-9):
    """gNFW Eqn 15 and 16 Wright & Brainerd 2000.

    Analytical normalized shear/deltaSigma profile.

    Args:
        x (array): Rp/Rs where Rs is the scale radius, Rs = R200/c
    """
    res = np.zeros_like(x)
    ix = np.where(np.abs(x - 1) <= eps)[0]
    res[ix] = 10. / 3. + 4 * np.log(1 / 2.)

    ix = np.where(x <= 1 - eps)[0]
    res[ix] = g_less_than(x[ix])

    ix = np.where(x >= 1 + eps)[0]
    res[ix] = g_greater_than(x[ix])
    return res


def convert_m200_to_r200(m200, rho, odelta=200):
    rv = np.power(3 * m200 / (4.0 * np.pi * rho * odelta), 1.0 / 3.0)
    return rv
