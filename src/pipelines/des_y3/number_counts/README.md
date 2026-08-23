# Cluster number counts

This observable predicts the expected number of clusters in each observed
richness and observed-redshift bin. It is the simplest member of the DES Y3
observable family because there is no radial profile to evaluate.

## Definition

For richness bin $i$ and observed-redshift bin $j$,

$$
N_{ij} = \int dz\,d\ln M\,d\lambda_{\rm tr};
n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
S_j(z)S_i(\lambda_{\rm tr},z)
P_{\rm HOD}(\lambda_{\rm tr}\mid M,z).
$$

Here $n(M,z)$ is the halo mass function, $P_{\rm HOD}$ is the
true-richness distribution, $S_i$ maps true richness to observed richness,
and $S_j$ is the photometric-redshift selection. The integration limits and
kernel conventions are inherited from the maintained selection model.

