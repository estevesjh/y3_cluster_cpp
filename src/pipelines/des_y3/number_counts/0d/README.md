# Fixed-GL count backends (`0d`)

This folder holds every number-counts backend with **zero adaptive
integration dimensions** — all integrals are fixed Gauss--Legendre sums, so
everything here is fast and MCMC-viable. Two algorithms share the folder:
the redshift-contracted fast path (formerly `fast_mass`) and the explicit
fixed-GL Python reference (formerly the Python half of `full_ltmz`). The
adaptive explicit references live in [`../3d/`](../3d/README.md).

## Redshift-contracted fast path

(Formerly `fast_mass`.) Reproduces the production count calculation by
separating the redshift and selection contraction from the remaining mass
integral.

## Numerical definition

Define the selection factor at fixed mass and redshift,

$$
S_{ij}(M,z)=S_j(z)\int d\lambda_{\rm tr}
S_i(\lambda_{\rm tr},z)P_{\rm HOD}(\lambda_{\rm tr}\mid M,z).
$$

The production path tabulates $S_{ij}$ on a fixed $(\ln M,z)$ grid and
bilinearly interpolates it. The redshift contraction is then

$$
W_{ij}(M)=\int dz\;n(M,z)\frac{dV}{d\Omega dz}\Omega(z)S_{ij}(M,z),
$$

and the count is

$$
N_{ij}=\int d\ln M\;W_{ij}(M).
$$

The radial-series population moments use the same $W_{ij}(M)$, which is why
the shared fixed-GL weight builder is kept in `shared`.

## Common algorithm

1. Read the production $S_{ij}(\ln M,z)$ table and the fixed GL mass and
   redshift nodes.
2. Bilinearly interpolate $S_{ij}$ at every node, with the production clamp
   and zero-outside conventions.
3. Multiply by the halo mass function, volume element, and survey area.
4. Sum over redshift to form $W_{ij}(M)$.
5. Integrate or sum $W_{ij}(M)$ over the fixed mass nodes.

This is a computational re-expression of the production algorithm. It does
not remove the production table's interpolation error.

## Language implementations

| Language | Algorithm and source files | Status |
| --- | --- | --- |
| Python | `python/numcounts_fast_mass.py` uses `shared/datablock_models.py` and `MassZWeights`, a convention-exact replica of `SelGLCore`; `python/validate_fast_vs_production.py` is its validator. | Readable reference replica |
| C++ | `cpp/NumCountsFastMass.cc` is a thin DES Y3-labelled wrapper around the existing fixed-GL count core (physics in `cpp/num_counts_fast_mass_t.hh`). | Production identity path |
| CUDA | No CUDA backend is provided. A one-dimensional mass contraction does not justify a GPU kernel in the current pipeline. | Not implemented |

## Precision and cost

Pinned 12-bin fiducial measurements:

| Dims | Backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `0d` | Python (2-dim GL, S_ij tab) | 5 ms/sample | $7.6\times10^{-4}$; also $2.4\times10^{-15}$ vs production (separate baseline) |
| `0d` | C++ (2-dim GL, S_ij tab) | 6 ms/sample | $7.6\times10^{-4}$; also identity with production (separate baseline) |

The residual against the adaptive reference is dominated by the production
\(S_{ij}\) tabulation and interpolation, not by a difference between the
Python and C++ implementations.

## Explicit fixed-GL Python reference

(Formerly the fixed-GL Python half of `full_ltmz`.) Evaluates the full
explicit \((\lambda_{\rm tr}, \ln M, z)\) composition — selection kernels at
the quadrature nodes, no \(S_{ij}\) tabulation — on fixed GL grids (96 mass,
64 redshift, 32 true-richness nodes), so it carries zero adaptive dimensions
and belongs here. Its adaptive C++/CUDA twins are the
[`../3d/`](../3d/README.md) references.

| Dims | Module | Output | Cost / accuracy |
| --- | --- | --- | --- |
| `0d` | `python/numcounts_full_ltmz.py` (3-dim GL; kernels imported from `shared/sel_kernels.py`, adaptive mass reference in `shared/full_ltmz_core.py`) | `numcounts_full_ltmz/vals` | 83 ms/sample; \(3.5\times10^{-5}\) vs the adaptive mass reference |
| `0d` | `python/validate_explicit_vs_production.py` (validator) | — | explicit fixed-GL vs production |
