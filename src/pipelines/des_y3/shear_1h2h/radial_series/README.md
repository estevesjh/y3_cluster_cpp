# Moment-expanded radial shear

This strategy replaces the mass-dependent radial profile evaluation with an
offline expansion around the population mean of the scale-radius coordinate.
It is a candidate speed approximation for one-halo shear, not a general
profile emulator.

## Strong model limitation

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

## Mathematical construction

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

## Common algorithm

1. Generate the fixed centred and miscentred unit profiles on logarithmic
   dimensionless grids.
2. Differentiate along the common scale-radius direction with a high-order
   stencil and divide by the appropriate factorial.
3. Average the single-offset profile over the fixed gamma miscentering kernel.
4. Save \(U_0\ldots U_3\) as versioned NumPy and text arrays.
5. At runtime, build the exact fixed-GL redshift weight, calculate \(N\),
   \(\bar y\), \(\mu_2\), and \(\mu_3\), then interpolate the arrays and
   restore \(A_0(\bar y)\).

## Language implementations

| Language | Algorithm and source files | Status |
| --- | --- | --- |
| Python | `python/generate_radial_series_tables.py` generates the arrays; `python/nfw_profile_family.py` defines the fixed profile; `python/shear1h_radial_series.py` evaluates the series; `python/validate_radial_series.py` checks derivatives and truncation. | Reference generator/evaluator |
| C++ | `cpp/shear1h_radial_series_t.hh` loads the text arrays and performs GSL bilinear/linear interpolation; `cpp/Shear1hRadialSeries.cc` is the module driver. | Candidate CPU evaluator |
| CUDA | No implementation. The runtime work is a few table lookups per radius and is not currently a useful GPU target. | Not implemented |

The table data are documented in
[`data/radial_series/README.md`](../../../../../../data/radial_series/README.md).

## Precision and cost

Pinned 12-bin × 10-radius fiducial measurements:

| Backend | Cost | Comparison |
| --- | ---: | --- |
| Python | 6--7 ms/sample | \(3.7\times10^{-3}\) total for the fixed-profile fiducial |
| C++ | 7 ms/sample | Same \(3.7\times10^{-3}\) plus \(1.6\times10^{-4}\) interpolation difference vs Python |

These numbers measure internal consistency and the fixed-profile approximation.
They do not certify agreement with the production profile that uses a varying
concentration relation. The known raw-(\Delta\Sigma) mismatch is documented in
[`docs/known_issues/radial_series_vs_full_ltmz_defect.md`](../../../../../../docs/known_issues/radial_series_vs_full_ltmz_defect.md).
