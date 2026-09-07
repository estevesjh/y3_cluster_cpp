"""Mass-concentration relations.

Consolidates the concentration functions previously scattered across
``y3_buzzard/haloModel.py`` (``child18_mass_concentration``,
``peakHeight_nonLinearMass``, ``duffy_concentration_relation``) and
``y3_buzzard/mass_concentration.py`` (``c_from_m200``, the hydro_mc-based
M200m route). ``y3_buzzard/massconcen.py`` is a byte-for-byte duplicate
of the latter (plus a dead, unconditionally-raising
``c_from_m200_ragagnin``) and is not ported.

The ``y3_buzzard`` originals stay in place — this is a copy into the
canonical shared location, not a move; see ``README.md`` in this
directory for why.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from . import hydro_mc


def child18_mass_concentration(M200c, z, halo_sample='stacked_nfw'):
    """The concentration mass relation model of Child et al 2018.

    We assume that this is an universal relation that does not depend on cosmology

    Parameters
    -----------------------------------------------------------------------------------------------
    M200c: array_like
        Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
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
    return 10**(12.5 - 1.5 * z)


def duffy_concentration_relation(m_h, z_eff=0.4):
    a_eff = 1 / (1 + z_eff)
    m_h_pivot = 2e12
    return 7.85 * np.power(m_h / m_h_pivot, -0.081) * np.power(a_eff, 0.71)


# m200 is mass M200m in linear units of solar mass/h^-1
def c_from_m200(m200, z, omega_m, omega_b, sigma8, h0, mstar, mstar_z):
    """Child et al. 2018 eq 18, entered from M200m via the hydro_mc M200m->M200c conversion.

    Uses parameters from the "individual, all" row of Child+18 Table 1
    (distinct from ``child18_mass_concentration``'s default
    ``stacked_nfw`` row), and a caller-supplied ``mstar(z)`` table (eqs
    13-17 of Child+18) rather than the ``peakHeight_nonLinearMass``
    approximation above.
    """
    A = 3.44
    b = 430.49
    m = -0.10
    co = 3.19

    interp = interp1d(mstar_z, mstar)
    Mstar = interp(z)

    # eq 18 demands M200c, not M200m. Convert via the mass-definition
    # relations of Ragagnin, Saro, Singh, & Dolag 2021
    # (https://github.com/aragagnin/hydro_mc).
    M_200c = hydro_mc.mass_from_mm_relation(
        "200m",
        "200c",
        M=m200,
        a=1.0 / (1 + z),
        omega_m=omega_m,
        omega_b=omega_b,
        sigma8=sigma8,
        h0=h0,
    )
    # note M_200c should be strictly less then M_200m

    mmb = M_200c / Mstar / b
    c200 = A * (mmb**m * (1 + mmb) ** (-1 * m) - 1) + co
    return c200
