#!/usr/bin/env python3
"""Hankel-space truth pins for test/miscentering_operator.test.{cc,py}.

Gamma-miscentred excess surface density of an analytic NFW, computed without
any Sigma -> DeltaSigma aperture integral:

    DeltaSigma_mis(R) = int_0^inf k dk / 2pi  rho~(k) P~(k) J_2(kR),

rho~(k) = 4 pi rho_s r_s^3 [sin q (pi/2 - Si q) - cos q Ci q], q = k r_s (the
3D NFW transform = 2D transform of Sigma, projection-slice theorem), and
P~(k) = (1 + k^2 sigma^2)^(-3/2), the transform of R_s/sigma^2 exp(-R_s/sigma).
With P~ = 1 the same integral reproduces the analytic centred DeltaSigma_NFW
to ~1e-12 (checked below), which validates the transform machinery.

Fiducial (M_sun/h, h^-1 Mpc, M_sun h/pc^2): M200m = 2e14, c = 4,
rho_m = 0.3 * 2.77533742639e11, sigma = 0.17 (52.5/100)^0.2.
Run: python test/make_miscentering_nfw_truth_pins.py  (needs mpmath)
"""
import mpmath as mp

mp.mp.dps = 30
M, C = mp.mpf("2e14"), mp.mpf(4)
RHO_M = mp.mpf("0.3") * mp.mpf("2.77533742639e11")
SIGMA = mp.mpf("0.17") * (mp.mpf("52.5") / 100) ** mp.mpf("0.2")
R200 = mp.cbrt(3 * M / (800 * mp.pi * RHO_M))
RS = R200 / C
RHO_S = mp.mpf(200) / 3 * C**3 / (mp.log(1 + C) - C / (1 + C)) * RHO_M
UNIT = mp.mpf("1e-12")
RADII = ["0.03", "0.1", "0.2", "0.5848", "1.0", "2.44521", "5.0"]


def rho_k(k):
    q = k * RS
    return 4 * mp.pi * RHO_S * RS**3 * (mp.sin(q) * (mp.pi / 2 - mp.si(q)) - mp.cos(q) * mp.ci(q))


def hankel(R, sigma=None):
    pk = (lambda k: 1) if sigma is None else (lambda k: (1 + (k * sigma) ** 2) ** mp.mpf(-1.5))
    val = mp.quadosc(lambda k: k * rho_k(k) * pk(k) * mp.besselj(2, k * R), [0, mp.inf], omega=R)
    return val / (2 * mp.pi) * UNIT


def dsigma_nfw(R):
    x = R / RS
    h = (2 / mp.sqrt(1 - x * x) * mp.atanh(mp.sqrt((1 - x) / (1 + x))) if x < 1
         else 2 / mp.sqrt(x * x - 1) * mp.atan(mp.sqrt((x - 1) / (x + 1))))
    F = (1 - h) / (x * x - 1)
    g = mp.log(x / 2) + h
    return 2 * RHO_S * RS * (2 * g / (x * x) - F) * UNIT


if __name__ == "__main__":
    print(f"// sigma = {mp.nstr(SIGMA, 15)}, r_s = {mp.nstr(RS, 15)}")
    for r in RADII:
        R = mp.mpf(r)
        cen = hankel(R)
        assert abs(cen / dsigma_nfw(R) - 1) < 1e-10, (r, cen, dsigma_nfw(R))
        print(f"  {{{r}, {mp.nstr(hankel(R, SIGMA), 15)}}},")
