# Offline unit-profile tables

This directory contains versioned tables for a reusable radial-series
factorization. The method does not depend on the physical interpretation of
the second coordinate. It only requires a fixed dimensionless profile
$u(x,x_{\rm mis})$ whose coordinates vary with mass through a common scale
radius.

## Important model limitation

The current table is **not** a general NFW radial profile. It is generated
for one fixed profile family with

$$
c(M,z)=4.
$$

Thus it assumes no concentration--mass evolution and no
concentration--redshift evolution. In particular,

$$
r_s(M)=\frac{r_{200}(M;\rho_{\rm crit})}{4}.
$$

The scale radius still has the explicit $r_s\propto M^{1/3}$ dependence from
the halo mass. What is frozen is the concentration and the dimensionless
profile shape: $u$ is reused for every mass and redshift. The consumer still
performs the redshift contraction of the population weights, but that does
**not** restore a production $c(M,z)$ relation.

This table is therefore an optional speed approximation for the fixed-shape
family. It is not required by the physical model. If the production profile
uses a varying concentration, direct radial evaluation is the safe choice;
an offline table would require an additional concentration-dependent
representation and independent validation.

The tables are generated once, committed to the repository, and loaded when
the consumer is constructed. They are not regenerated during an MCMC sample.

## Files and code

| File or path | Role |
| --- | --- |
| [radial_series_nfw_mis_gamma_v1.npz](radial_series_nfw_mis_gamma_v1.npz) | Primary binary table of the stored coefficient functions. |
| [radial_series_nfw_mis_gamma_v1.json](radial_series_nfw_mis_gamma_v1.json) | Metadata, fixed profile conventions, generator provenance, source checksums, and validation results. |
| radial_series_nfw_mis_gamma_v1_*.txt | Human-readable exports of the axes and coefficient arrays. |
| [generate_radial_series_tables.py](../../src/pipelines/des_y3/observables/shear_1h2h/radial_series/python/generate_radial_series_tables.py) | Generates a table from the fixed $c=4$ profile definition; it is not a general concentration-aware generator. |
| [validate_radial_series.py](../../src/pipelines/des_y3/observables/shear_1h2h/radial_series/python/validate_radial_series.py) | Checks the generated coefficients and interpolation. |
| [shear1h_radial_series.py](../../src/pipelines/des_y3/observables/shear_1h2h/radial_series/python/shear1h_radial_series.py) | Current consumer of the stored coefficients. |
| [src/pipelines/des_y3/README.md](../../src/pipelines/des_y3/README.md) | General design rule for offline unit-profile tables (the retired `docs/module_reorganization_plan.md` proposal, now folded in). |

The labels cen and mis in the filenames identify the two coefficient sets
stored by this particular table. They are data labels; the factorization below
only requires a fixed profile function and two mass-dependent coordinates.

## Factorized profile

Let

$$
 y=\ln r_s(M),
 \qquad
 x=\frac{R}{r_s(M)}=R e^{-y},
 \qquad
 x_{\rm mis}=\frac{\tau}{r_s(M)}=\tau e^{-y},
$$

where $R$ is the first physical coordinate and $\tau$ is a second physical
coordinate. The name $x_{\rm mis}$ is retained because it is the name used by
this table; the method does not require $\tau$ to represent a particular
physical effect.

The profile is factored as

$$
\Phi(R,y,\tau;\boldsymbol\vartheta)
=A_{\rm sample}(\boldsymbol\vartheta)
 A_0(y)
 u\left(x,x_{\rm mis}\right).
$$

Here:

- $u(x,x_{\rm mis})$ is the fixed, dimensionless profile shape tabulated by
  this directory;
- $A_0(y)$ is the fixed scale-radius normalization used when the table was
  generated;
- $A_{\rm sample}(\boldsymbol\vartheta)$ is any sample-dependent multiplier
  that is independent of $y$ and is restored by the consumer.

$A_{\rm sample}$ is not assigned a universal value here. It is not part of
$U_\ell$ and it is not a miscentering parameter. If a sample-dependent
parameter changes the shape of $u$, the existing table cannot be reused.

For the current table, the fixed normalization is

$$
A_0(y)=2e^y\,\delta_c\,\rho_{\rm crit}\,10^{-12},
$$

but the factorization rule itself is not limited to this particular
normalization.

## Coefficient definitions

The two coordinates move together when the common scale radius changes, so the
operator used by this table is

$$
L
=\frac{\partial}{\partial\ln x}
+\frac{\partial}{\partial\ln x_{\rm mis}}.
$$

The coefficient functions are defined by the scale-radius expansion

$$
U_\ell(x,x_{\rm mis})
=\frac{1}{\ell!\,A_0(y)}
\frac{\partial^\ell}{\partial y^\ell}
\left[A_0(y)u\left(R e^{-y},\tau e^{-y}\right)\right].
$$

For the current table, $A_0(y)\propto e^y$, so the stored coefficients can
be written directly as

$$
U_0=u,
$$

$$
U_1=(1-L)u,
$$

$$
U_2=\frac{1}{2}\left(1-2L+L^2\right)u,
$$

$$
U_3=\frac{1}{6}\left(1-3L+3L^2-L^3\right)u.
$$

Here $L^2$ and $L^3$ mean repeated application of the same operator. These
are definitions of the stored coefficient functions; no population integral
or physical mixture is part of this table definition.

## Stored arrays and ranges

| Array or metadata key | Shape | Meaning |
| --- | --- | --- |
| lnx | $(619,)$ | Grid for $\ln x$. |
| lnxm | $(306,)$ | Grid for $\ln x_{\rm mis}$. |
| $U_\ell^{\rm mis}$ | $(619,306)$ for $\ell=0,\ldots,3$ | Coefficients on the two-coordinate grid. |
| $U_\ell^{\rm cen}$ | $(619,)$ for $\ell=0,\ldots,3$ | Coefficients on the one-coordinate grid used by this table variant. |
| meta_json | scalar | Serialized copy of the metadata in the JSON sidecar. |

The axes are

$$
 x\in[9.8\times10^{-4},5.0\times10^3],
 \qquad
 x_{\rm mis}\in[9.8\times10^{-3},20].
$$

Queries outside the stored ranges are clamped by the interpolation routines.
The metadata records the fixed profile conventions, units, radial domain,
generator provenance, source checksums, numerical scheme, and validation
results.

## Consumer contract

A consumer supplies the current sample's scale-radius coordinate and evaluates
the stored functions at

$$
\ln x=\ln R-y,
\qquad
\ln x_{\rm mis}=\ln\tau-y.
$$

It then restores $A_{\rm sample}A_0(y)$ and applies whatever population
weighting, moment truncation, or physical combination belongs to its own
observable. Those operations are deliberately outside this data definition.

The coefficient $U_1$ is stored for validation and for consumers that need the
first-order term. A consumer may truncate the expansion at any supported
order; the appropriate accuracy is part of that consumer's validation.

## Validation and limitations

The generated coefficients are compared with the source profile and with
independent derivative calculations. The source lookup values contain roughly
$10^{-5}$--$3\times10^{-3}$ point-to-point noise in $\ln u$, so the generator
constructs the fixed profile smoothly before producing derivatives. The source
values are used for fidelity checks, not differentiated directly.

| Check | Result |
| --- | --- |
| $U_0$ versus the source profile | Median $|\Delta\ln u|=1.7\times10^{-5}$; maximum $3.8\times10^{-4}$ over the physical window. |
| One-coordinate coefficients versus high-precision derivatives | $\le1.6\times10^{-10}$ during generation; $\le1.1\times10^{-8}$ including runtime interpolation. |
| Two-coordinate $U_1$--$U_3$ versus an independent derivative check | Median $\le10^{-8}$; maximum $\le5\times10^{-5}$. |

The complete validation report is [validate_radial_series.py](../../src/pipelines/des_y3/observables/shear_1h2h/radial_series/python/validate_radial_series.py).

## Versioning and reuse

A change to the fixed profile shape, coordinate convention, normalization
convention, or supported axis range requires a new versioned table and a new
generator run. Do not overwrite an existing version in place.

A table may be reused across samples only when the dimensionless profile shape
$u(x,x_{\rm mis})$ and the fixed normalization convention $A_0(y)$ remain
unchanged. Sample-dependent amplitudes, coordinates, and downstream weights
belong to the consumer.

For this table, “remain unchanged” includes the strong assumption
$c(M,z)=4$. A consumer that needs mass- or redshift-dependent concentration
must not silently reuse these arrays. It should evaluate the radial profile
directly or use a separately derived, concentration-aware approximation.
