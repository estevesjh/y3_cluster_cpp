# Redshift-contracted count integration

This strategy reproduces the production count calculation by separating the
redshift and selection contraction from the remaining mass integral.

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
| Python | `python/numcounts_fast_mass.py` uses `shared/datablock_models.py` and `MassZWeights`, a convention-exact replica of `SelGLCore`. | Readable reference replica |
| C++ | `cpp/NumCountsFastMass.cc` is a thin DES Y3-labelled wrapper around the existing fixed-GL count core. | Production identity path |
| CUDA | No CUDA backend is provided. A one-dimensional mass contraction does not justify a GPU kernel in the current pipeline. | Not implemented |

## Precision and cost

Pinned 12-bin fiducial measurements:

| Backend | Cost | Comparison |
| --- | ---: | --- |
| Python | 5 ms/sample | $7.6\times10^{-4}$ vs the adaptive full-ltmz reference; $2.4\times10^{-15}$ vs production |
| C++ | 6 ms/sample | $7.6\times10^{-4}$ vs the adaptive reference; identity with production |

The residual against the adaptive reference is dominated by the production
\(S_{ij}\) tabulation and interpolation, not by a difference between the
Python and C++ implementations.
