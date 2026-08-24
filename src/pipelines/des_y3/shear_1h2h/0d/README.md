# Fixed-GL and tabulated shear backends (`0d`)

This folder holds every shear_1h2h backend with **zero adaptive integration
dimensions**: all integrals are fixed Gauss--Legendre sums or offline table
lookups, so every backend here is fast and MCMC-viable. Four algorithms
share the folder (formerly the separate `fast_mass`, `radial_series`, and
the fixed-GL half of `full_ltmz` strategies); the adaptive references they
validate against live in [`../3d/`](../3d/README.md).

| Algorithm | Backends | Modules |
| --- | --- | --- |
| [1h z-contracted GL](#one-halo-z-contracted-gl) | Python, C++ | `shear1h_fast_mass.py`, `Shear1hFastMass.cc` |
| [Max model z-resolved GL](#traditional-1h2h-max-model-z-resolved-gl) | Python, C++, CUDA | `shear1h2h_max.py`, `Shear1h2hMax.cc`, `Shear1h2hMaxGpu.cu` |
| [Explicit fixed-GL references](#explicit-fixed-gl-python-references) | Python | `shear1h_full_ltmz.py`, `shear1h2h_max_full_ltmz.py` |
| [Moment-expanded radial series](#moment-expanded-radial-series) | Python, C++ | `shear1h_radial_series.py`, `Shear1hRadialSeries.cc` |

The C++ modules all build into one binary dir:
`release-build/src/modules/des_y3_shear1h_0d_cpp/` (three `.so`); the CUDA
max-model backend builds into `des_y3_shear1h_0d_cuda/`.

---

## One-halo z-contracted GL

(Formerly `fast_mass`, one-halo half.) Evaluates the exact fixed-GL redshift
contraction first and then performs the mass sum against the production
one-halo profile. It is the maintained fast path for one-halo shear. The
traditional max model cannot use this contraction — its two-halo term is
z-dependent — see the next section.

At fixed mass, define

$$
S_{ij}(M,z)=S_j(z)\int d\lambda_{\rm tr}
S_i(\lambda_{\rm tr},z)P_{\rm HOD}(\lambda_{\rm tr}\mid M,z).
$$

The lensing weight is

$$
W_{ij}(M)=\int dz\;n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
\Sigma_{\rm crit}^{-1}(z)S_{ij}(M,z),
$$

and the one-halo result is

$$
O_{ij}(R)=\int d\ln M\;W_{ij}(M)
\left[(1-f_{\rm mis})\Delta\Sigma_{\rm cen}(R,M)
 +f_{\rm mis}\Delta\Sigma_{\rm mis}(R,M)\right].
$$

The production fast path evaluates or interpolates \(S_{ij}\) on its fixed
grid, then performs this mass contraction directly on the two
\(\Delta\Sigma\) components.

Algorithm:

1. Build the fixed GL mass/redshift nodes and read the production selection
   table.
2. Interpolate the selection factor at each node using the production
   bilinear/clamp convention.
3. Multiply by the HMF, volume element, survey area, inverse critical density,
   and the richness/redshift weights.
4. Contract over redshift to obtain \(W_{ij}(M)\).
5. Evaluate the production miscentred mixture at every requested radius and
   contract over mass.

| Language | Algorithm and source files | Status |
| --- | --- | --- |
| Python | `python/shear1h_fast_mass.py` uses `shared/datablock_models.py` and `shared/lensing_profiles.py`; `python/validate_fast_vs_production.py` is its validator. | Readable replica |
| C++ | `cpp/Shear1hFastMass.cc` is the one-halo identity wrapper (physics in `cpp/shear1h_fast_mass_t.hh`), composing the immutable GL core. | Maintained CPU backend |

| Dims | Backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `0d` | One-halo Python (1-dim GL, z contracted) | 74 ms/sample | \(8.4\times10^{-4}\); also \(3.1\times10^{-15}\) vs production (separate baseline) |
| `0d` | One-halo C++ (1-dim GL, z contracted) | 9 ms/sample | \(8.4\times10^{-4}\); also production identity (separate baseline) |

## Traditional 1h+2h max model, z-resolved GL

(Formerly the max-model half of `fast_mass`.) Unlike the one-halo path, the
redshift integral cannot be contracted past the profile: the biased two-halo
term \(b(M,z)\,\Delta\Sigma_{hh}(R,z)\) is z-dependent and the max is
nonlinear, so the contraction keeps the z-resolved weight — a double fixed-GL
(lnM, z) sum per (bin, R), still zero adaptive dimensions.

$$
\Delta\Sigma_{\rm max}(R,M,z)=
\max\left[\Delta\Sigma_{1h}(R,M,z),
b(M,z)\Delta\Sigma_{hh}(R,z)\right],
$$

$$
O_{ij}^{\rm max}(R)=\int dz\,d\ln M\;W_{ij}(M,z)
\Sigma_{\rm crit}^{-1}(z)\Delta\Sigma_{\rm max}(R,M,z),
$$

with \(W_{ij}(M,z)\) the z-RESOLVED selection weight (same term composition
as the one-halo GL core, without the z sum) and \(\Delta\Sigma_{1h}\) the
production miscentred mixture. The S_ij tabulation is still the fast-path
hallmark.

| Language | Algorithm and source files | Status |
| --- | --- | --- |
| Python | `python/shear1h2h_max.py` is the z-resolved max model; `python/validate_shear1h2h_max.py` is its validator. | Readable replica |
| C++ | `cpp/Shear1h2hMax.cc` (physics in `cpp/shear1h2h_max_t.hh`) builds the z-resolved weight on fixed GL nodes and contracts (lnM, z) per (bin, R). | Maintained CPU backend |
| CUDA | `cuda/Shear1h2hMaxGpu.cu` (physics in `cuda/shear1h2h_max_gpu_t.cuh`) contracts the heavy miscentred-NFW part on the device; host code supplies the remaining tables and weights. | GPU acceleration |

| Dims | Backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `0d` | Max model C++ (2-dim GL, z-resolved) | 11 ms/sample | \(8.3\times10^{-4}\); also \(6.0\times10^{-15}\) vs Python twin (separate baseline) |
| `0d` | Max model CUDA / A100 (2-dim GL, z-resolved) | 8 ms/sample | \(6.4\times10^{-15}\) vs C++ twin (separate baseline); vs-3d inherited through the C++ row |

The explicit references this algorithm validates against live in
[`../3d/`](../3d/README.md) (`Shear1h2hMaxFullLtmz` C++) and in the
[explicit fixed-GL section](#explicit-fixed-gl-python-references) below
(`shear1h2h_max_full_ltmz.py`).

The max-model two-halo term has a separate known numerical/debugging issue;
see `docs/known_issues/dsigma_hh_debug_flag.md` and the validation notes before using it as
a scientific reference.

## Explicit fixed-GL Python references

(Formerly the fixed-GL Python half of `full_ltmz`.) These evaluate the full
explicit \((\lambda_{\rm tr}, \ln M, z)\) composition — every selection
kernel at the quadrature nodes, no \(S_{ij}\) tabulation — but on fixed GL
grids (96 mass, 64 redshift, 32 true-richness nodes), so they carry zero
adaptive dimensions and belong here. Their adaptive C++/CUDA twins are the
[`../3d/`](../3d/README.md) references.

| Dims | Module | Observable | Output | Cost / accuracy |
| --- | --- | --- | --- | --- |
| `0d` | `python/shear1h_full_ltmz.py` (3-dim GL) | one-halo shear | `shear1h_full_ltmz/vals` | 149 ms/sample; \(4.9\times10^{-5}\) vs the adaptive reference |
| `0d` | `python/shear1h2h_max_full_ltmz.py` (3-dim GL) | 1h+2h max model | `shear1h2h_max_full_ltmz/vals` | z-RESOLVED weight (`full_ltmz_mass_z_weights`) × `MaxMixtureProfile`; the explicit reference the max-model GL backends validate against |
| `0d` | `python/validate_explicit_vs_production.py` | validator | — | explicit fixed-GL vs production |

Both max/1h explicit modules read `miscentering/{f_mis,tau_mis}` strictly —
no in-code default fallback — and the max module requires
`compute_lensing_2h = T` (dSigma_hh NaNs sanitized to 0, exact under the max).

## Moment-expanded radial series

(Formerly `radial_series`; no runtime integration — offline tables plus
population moments.) Replaces the mass-dependent radial profile evaluation
with an offline expansion around the population mean of the scale-radius
coordinate. It is a candidate speed approximation for one-halo shear, not a
general profile emulator.

### Strong model limitation

The committed table uses a fixed NFW concentration,

$$
c(M,z)=4.
$$

There is no concentration--mass or concentration--redshift evolution in the
tabulated dimensionless profile. The scale radius still varies with mass
through \(r_s\propto M^{1/3}\), but a production \(c(M,z)\) relation changes
both the amplitude and radial shape. Exact redshift contraction of the
population weights does not restore that missing dependence. Direct radial
evaluation is the correct choice when the production profile is required.

### Mathematical construction

Let

$$
y=\ln r_s(M),\qquad x=R e^{-y},\qquad
x_{\rm mis}=\tau e^{-y},
$$

and write the fixed profile as

$$
\Delta\Sigma(R,y,\tau)=A_0(y)u(x,x_{\rm mis}),
\qquad A_0(y)\propto e^y.
$$

For a population with mean \(\bar y\), define

$$
U_\ell(x,x_{\rm mis})=
\frac{1}{\ell!A_0(y)}
\left.\frac{\partial^\ell}{\partial s^\ell}
\left[A_0(y+s)u\left(Re^{-(y+s)},\tau e^{-(y+s)}\right)\right]
\right|_{s=0}.
$$

Because \(A_0\propto e^y\), only the two dimensionless coordinates are needed.
With

$$
L=\frac{\partial}{\partial\ln x}
 +\frac{\partial}{\partial\ln x_{\rm mis}},
$$

the stored functions are

$$
U_0=u,
\qquad U_1=(1-L)u,
$$

$$
U_2=\frac12(1-2L+L^2)u,
\qquad
U_3=\frac16(1-3L+3L^2-L^3)u.
$$

The mass integral is approximated using population moments,

$$
O(R)\approx N A_0(\bar y)\left[
U_0+\mu_2U_2+\mu_3U_3\right],
$$

with \(U_0\), \(U_2\), and \(U_3\) interpolated at
\((\ln R-\bar y,\ln\tau-\bar y)\). \(U_1\) is retained for validation; its
coefficient vanishes for the centered expansion used here.

### Algorithm

1. Generate the fixed centred and miscentred unit profiles on logarithmic
   dimensionless grids.
2. Differentiate along the common scale-radius direction with a high-order
   stencil and divide by the appropriate factorial.
3. Average the single-offset profile over the fixed gamma miscentering kernel.
4. Save \(U_0\ldots U_3\) as versioned NumPy and text arrays.
5. At runtime, build the exact fixed-GL redshift weight, calculate \(N\),
   \(\bar y\), \(\mu_2\), and \(\mu_3\), then interpolate the arrays and
   restore \(A_0(\bar y)\).

| Language | Algorithm and source files | Status |
| --- | --- | --- |
| Python | `python/generate_radial_series_tables.py` generates the arrays; `python/nfw_profile_family.py` defines the fixed profile; `python/shear1h_radial_series.py` evaluates the series; `python/validate_radial_series.py` checks derivatives and truncation. | Reference generator/evaluator |
| C++ | `cpp/shear1h_radial_series_t.hh` loads the text arrays and performs GSL bilinear/linear interpolation; `cpp/Shear1hRadialSeries.cc` is the module driver. | Candidate CPU evaluator |
| CUDA | No implementation. The runtime work is a few table lookups per radius and is not currently a useful GPU target. | Not implemented |

The table data are documented in
[`data/radial_series/README.md`](../../../../../data/radial_series/README.md).

| Dims | Backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `0d` | Python (tables + moments) | 6--7 ms/sample | 56--86% (known fixed-c=4 defect; see link below); \(3.7\times10^{-3}\) internal fixed-profile consistency (separate baseline) |
| `0d` | C++ (tables + moments) | 7 ms/sample | Same vs-3d defect; \(3.7\times10^{-3}\) plus \(1.6\times10^{-4}\) interpolation difference vs Python (separate baselines) |

These numbers measure internal consistency and the fixed-profile approximation.
They do not certify agreement with the production profile that uses a varying
concentration relation. The known raw-\(\Delta\Sigma\) mismatch is documented in
[`docs/known_issues/radial_series_vs_full_ltmz_defect.md`](../../../../../docs/known_issues/radial_series_vs_full_ltmz_defect.md).
