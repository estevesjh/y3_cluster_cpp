# Cluster number counts

This observable predicts the expected number of clusters in each observed
richness and observed-redshift bin. It is the simplest member of the DES Y3
observable family because there is no radial profile to evaluate.

## Definition

For richness bin \(i\) and observed-redshift bin \(j\),

$$
N_{ij} = \int dz\,d\ln M\,d\lambda_{\rm tr};
n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
S_j(z)S_i(\lambda_{\rm tr},z)
P_{\rm HOD}(\lambda_{\rm tr}\mid M,z).
$$

Here \(n(M,z)\) is the halo mass function, \(P_{\rm HOD}\) is the
true-richness distribution, \(S_i\) maps true richness to observed richness,
and \(S_j\) is the photometric-redshift selection. The integration limits and
kernel conventions are inherited from the maintained selection model.

## Strategies

| Strategy | Role | Implementations |
| --- | --- | --- |
| [`full_ltmz`](full_ltmz/README.md) | Independent reference: keeps true richness, redshift, and mass explicit | Python, C++, CUDA |
| [`fast_mass`](fast_mass/README.md) | Production-speed contraction: tabulates the selection weight, then integrates over mass | Python, C++ |
| `radial_series` | Not applicable: counts have no radial operator | — |

Read the strategy README before entering a language directory. The language
directories contain code and build files, not separate scientific contracts.
