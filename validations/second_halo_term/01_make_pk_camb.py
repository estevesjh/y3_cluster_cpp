#!/usr/bin/env python
"""Generate the fiducial P(k, z) with pycamb — the single shared P(k) input.

Run with the conda pipeline env (camb 1.4.0):
  /opt/homebrew/Caskroom/miniforge/base/envs/y3cl_je_macos/bin/python 01_make_pk_camb.py

Fiducial cosmology = the checked-in dump's cosmological_parameters section.
sigma8 is matched by rescaling As (exact to linear order; iterated).

Grids are matched to the dump's matter_power_lin grids so pinned test
benchmarks stay comparable: z = 50 pts in [0, 4], k_h = 506 pts in
[1e-4, 50] h/Mpc.

Output: outputs/pk_camb.npz with unit-ledger strings. Plain float64 arrays
only (safe across numpy 1.26 <-> 2.4); loaders must use allow_pickle=False.

Also records the dump-vs-CAMB linear P(k) consistency metric (the dump's
P(k) comes from the cp_camb emulator, not CAMB itself).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DUMP = os.path.join(REPO, "docs", "figs", "real_pipeline_extract_prj2h_output")
OUT = os.path.join(HERE, "outputs", "pk_camb.npz")

# fiducial (dump cosmological_parameters/values.txt)
H0 = 67.66
OMBH2 = 0.022420145751
OMCH2 = 0.11997421699944
NS = 0.9665
TAU = 0.0544
SIGMA8_TARGET = 0.8238
OMEGA_M = 0.311049
H = H0 / 100.0


def load_dump_grid():
    z = np.loadtxt(os.path.join(DUMP, "matter_power_lin", "z.txt"))
    k_h = np.loadtxt(os.path.join(DUMP, "matter_power_lin", "k_h.txt"))
    p_k = np.loadtxt(os.path.join(DUMP, "matter_power_lin", "p_k.txt"))
    p_k = p_k.reshape(z.size, k_h.size)  # _cosmosis_order: z rows, k cols
    return z, k_h, p_k


def camb_pk(As, z, k_h):
    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=OMBH2, omch2=OMCH2, mnu=0.0,
                       num_massive_neutrinos=0, omk=0.0, tau=TAU)
    pars.InitPower.set_params(As=As, ns=NS)
    # pin the nonlinear model: Takahashi (2012) halofit, the paper's
    # prescription for xi_NL (CAMB's default is mead2020/HMcode)
    pars.NonLinearModel.set_params(halofit_version="takahashi")
    pars.set_matter_power(redshifts=[0.0], kmax=60.0 / H)
    res = camb.get_results(pars)
    s8 = float(res.get_sigma8_0())

    kw = dict(zmin=0.0, zmax=float(z.max()) + 0.1, nz_step=120,
              kmax=60.0 / H, k_hunit=True, hubble_units=True,
              extrap_kmax=200.0)
    pk_lin = camb.get_matter_power_interpolator(pars, nonlinear=False, **kw)
    pk_nl = camb.get_matter_power_interpolator(pars, nonlinear=True, **kw)
    p_lin = pk_lin.P(z, k_h, grid=True)   # (nz, nk), (Mpc/h)^3, k in h/Mpc
    p_nl = pk_nl.P(z, k_h, grid=True)
    return s8, p_lin, p_nl


def main():
    z, k_h, p_dump = load_dump_grid()
    print(f"dump grids: nz={z.size} z in [{z[0]:.3g}, {z[-1]:.3g}], "
          f"nk={k_h.size} k_h in [{k_h[0]:.3g}, {k_h[-1]:.3g}] h/Mpc")

    As = 2.0e-9
    for it in range(5):
        s8, p_lin, p_nl = camb_pk(As, z, k_h)
        print(f"  As iteration {it}: As={As:.6e} sigma8={s8:.6f}")
        if abs(s8 - SIGMA8_TARGET) < 1e-4:
            break
        As *= (SIGMA8_TARGET / s8) ** 2
    assert abs(s8 - SIGMA8_TARGET) < 1e-4, f"sigma8 iteration failed: {s8}"

    # dump-vs-CAMB consistency (linear, over the dump's own grid)
    ratio = p_dump / p_lin
    dev = np.abs(ratio - 1.0)
    print(f"dump(emulator) vs CAMB linear P(k): max|dev|={dev.max():.3%}, "
          f"median={np.median(dev):.3%}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(
        OUT,
        k_h=k_h, z=z, p_k_lin=p_lin, p_k_nl=p_nl,
        p_k_dump=p_dump,
        As=np.float64(As), sigma8=np.float64(s8),
        h=np.float64(H), omega_m=np.float64(OMEGA_M),
        dump_dev_max=np.float64(dev.max()),
        dump_dev_median=np.float64(np.median(dev)),
        units_k="h/Mpc (comoving)",
        units_p="(Mpc/h)^3 (comoving)",
        rho_convention="none: raw matter power",
    )
    print(f"wrote {OUT}")
    if dev.max() > 0.02:
        print("WARNING: dump emulator P(k) deviates from CAMB by more than 2% "
              "somewhere — pinned benchmarks generated from dump P(k) may "
              "need the tolerance headroom. Gate evaluated in 07_compare.py.")


if __name__ == "__main__":
    main()
