#!/usr/bin/env python
"""Import the miscentered-NFW DeltaSigma_mis "single" (fixed-offset) table
from CLensPy, fixing the cusp-interpolation defect CLensPy's
docs/miscentering_math.md Sec.~9.3 diagnoses in the shipped
table_1000_1e-03_5e+03_deltasigma_signed_single.txt (and, by the same
argument, in the "gamma" table -- see GitHub issue #6 and
review/08272026/issue6_rationale.md).

THE DEFECT (both repos' natural (ln x, ln xmis) axes)
------------------------------------------------------
DeltaSigma_mis(x, xmis) has a sign-changing ridge along x = xmis (the
aperture crosses the true halo center). On natural axes that ridge cuts
diagonally across grid cells; bilinear interpolation smooths/flips it near
the ridge (CLensPy measured 6.5 relative error, WRONG SIGN, at x=xmis=0.01
on natural axes vs 8e-10 on the fix below).

THE FIX: grid on (ln xmis, ln q), q = x/xmis
---------------------------------------------
Putting q on the axis makes the ridge the vertical line ln q = 0; CLensPy's
generator (tools/make_miscentering_table.py) forces that to be an EXACT
grid node (odd inner-tier node count), so no interpolation cell ever
straddles it.

WHY IMPORT RATHER THAN REGENERATE
-----------------------------------
CLensPy's src/clenspy/data/nfw_miscentering.npz is already the exact same
physics y3_cluster_cpp's "single" kernel uses (one FIXED miscentering
offset, no offset-magnitude averaging; dimensionless, Sigma0=1, r_s=1,
generated via cluster_toolkit with a converged, non-truncated radial
domain -- CLensPy's own generator docs report a `--tune` convergence scan,
not a fixed cutoff like the Rp_max=15 recipe issue #6 flags). Re-deriving
it from scratch would just reproduce the same numbers at 10x the effort
and risk of transcription error -- so this script imports the array data
directly instead.

NOT COVERED: the "gamma" kernel (population-averaged offset magnitude,
identified as Gamma(shape=2, scale=xmis) by calibration against the
shipped table -- see review/08272026/gamma_kernel_draft_generator.py.txt)
has no CLensPy equivalent and stays on the legacy natural-axis grid for
now; nfw_dsigma_mis.hh/.cuh branch on kernel_is_signed() to pick the axis
convention per kernel.

Run once; output is committed (data files), not regenerated at build/test
time.
"""
import os
import numpy as np

CLENSPY_NPZ = os.path.expanduser(
    "~/Documents/Dev/github/CLensPy/src/clenspy/data/nfw_miscentering.npz")
CLENSPY_COMMIT = "25ff8567c0100bd5acf16ade6b5f24996c42eb6e"  # 2026-08-27, read-only ref
HERE = os.path.dirname(os.path.abspath(__file__))
PROVENANCE = (f"imported from CLensPy src/clenspy/data/nfw_miscentering.npz "
              f"@ {CLENSPY_COMMIT[:8]} (2026-08-27); see "
              f"data/nfw_off_center/import_clenspy_single_table.py")


def main():
    d = np.load(CLENSPY_NPZ)
    lnxmis, lnq, ds = d["ln_x_mis"], d["ln_q"], d["ds_hat_mis"]
    assert lnq[np.searchsorted(lnq, 0.0)] == 0.0 or np.any(lnq == 0.0), (
        "ln q = 0 must be an exact node (the cusp-safety property)")
    print(f"xmis: {lnxmis.size} pts in [{np.exp(lnxmis[0]):.3e}, {np.exp(lnxmis[-1]):.3e}]")
    print(f"q:    {lnq.size} pts in [{np.exp(lnq[0]):.3e}, {np.exp(lnq[-1]):.3e}]")
    print(f"ds_hat_mis: shape {ds.shape}, {100*(ds < 0).mean():.1f}% negative "
          f"(the physical lobe at xmis > x)")

    np.savetxt(os.path.join(HERE, "table_ratio_logxmis.txt"), lnxmis,
               fmt="%.16e", header=PROVENANCE)
    np.savetxt(os.path.join(HERE, "table_ratio_logq.txt"), lnq,
               fmt="%.16e", header=PROVENANCE)
    np.savetxt(os.path.join(HERE, "table_ratio_deltasigma_signed_single.txt"),
               ds, fmt="%.10e", header=PROVENANCE)
    print("wrote table_ratio_{logxmis,logq,deltasigma_signed_single}.txt")


if __name__ == "__main__":
    main()
