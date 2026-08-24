# One-halo and traditional one-plus-two-halo shear

This observable contains the maintained miscentered one-halo model and the
optional traditional max model.

## Numerical definitions

The one-halo profile is a centered/miscentered mixture,

$$
\Delta\Sigma_{1h}(R,M,z)=
(1-f_{\rm mis})\Delta\Sigma_{\rm cen}(R,M,z)
+f_{\rm mis}\Delta\Sigma_{\rm mis}(R,M,z).
$$

The stacked excess surface density is

$$
O_{ij}(R)=\int dz\,d\ln M\,d\lambda_{\rm tr}\;
n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
S_j(z)S_i(\lambda_{\rm tr},z)
P_{\rm HOD}(\lambda_{\rm tr}\mid M,z)
\Delta\Sigma_{1h}(R,M,z).
$$

The `Sigma_crit^{-1}` factor is optional. Set it to unity when computing
`DeltaSigma`; including the lensing-geometry factor instead returns the
tangential shear `gamma_T`.

The optional traditional max model retains the redshift-dependent two-halo
term,

$$
\Delta\Sigma_{\rm max}(R,M,z)=
\max\left[\Delta\Sigma_{1h}(R,M,z),
b(M,z)\Delta\Sigma_{hh}(R,z)\right].
$$

The numerical implementations are documented in the strategy READMEs:
[`full_ltmz`](full_ltmz/README.md), [`fast_mass`](fast_mass/README.md), and
[`radial_series`](radial_series/README.md).

## Precision and cost

Pinned 12-bin by 10-radius fiducial measurements; cost is per sample.

| Method and backend | Cost | Comparison or status |
| --- | ---: | --- |
| [`full_ltmz`](full_ltmz/README.md), adaptive Python | 35 s | Reference; reported integration error at or below 1e-6 |
| [`full_ltmz`](full_ltmz/README.md), fixed GL Python | 149 ms | 4.9e-5 vs adaptive reference |
| [`full_ltmz`](full_ltmz/README.md), Cuhre C++ | 51 s | 3.3e-4 vs Python reference |
| [`full_ltmz`](full_ltmz/README.md), PAGANI CUDA/A100 | 32 s | 3.4e-4 vs C++ reference |
| [`fast_mass`](fast_mass/README.md), one-halo Python/C++ | 74 / 9 ms | 8.4e-4 vs adaptive; identity with production |
| [`fast_mass`](fast_mass/README.md), max C++/CUDA/A100 | 11 / 8 ms | 8.3e-4 vs adaptive; CUDA agrees with C++ to 6.4e-15 |
| [`radial_series`](radial_series/README.md), Python/C++ | 6--7 ms | Internal fixed-profile consistency only; not production accuracy |

The traditional max model is provisional because its two-halo input has a
separate `haloModel/dSigma_hh` data defect.

All backends read their lensing profile tables (`dSigma_nfw`, `dSigma_hh`,
`bias`, `concentration`) from the `haloModel` datablock section, whose
halo-model physics lives in the cosmology layer:
[`../../cosmology/halo_model.py`](../../cosmology/halo_model.py) (driven
through the CosmoSIS wrapper `y3_buzzard/halo_model_cosmosis.py`).
