# Halo Bias

`Python` (producer) · `y3_cluster_cpp` (`y3_buzzard/halo_model_cosmosis.py`, physics in `src/pipelines/cosmology/halo_model.py`) · `Halo population`

$b(M, z)$ quantifies how strongly haloes cluster relative to the matter
field. In this pipeline it is **not a module of its own**: the
`halo_model` module publishes it as a DataBlock grid, and three consumers
interpolate it.

## Script

- Producer: [`y3_buzzard/halo_model_cosmosis.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/y3_buzzard/halo_model_cosmosis.py)
  via class `biasModel` in
  [`src/pipelines/cosmology/halo_model.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/cosmology/halo_model.py)
  (path-stable original `y3_buzzard/haloModel.py`) and
  `cluster_toolkit.bias.bias_at_nu` — see {doc}`halo_model` for the module
  mechanics. Pinned against `cluster_toolkit` by
  `test/cosmology_package.test.py`.
- The retired C++ halo-bias type `HMB_t` (and the `tinker_bias` module) no
  longer exist; every C++ consumer reads the DataBlock grid through
  `Interp2D`.

## Numerical framework

Tinker et al. (2010) fitting function of the peak height,

$$b(\nu) = 1 - A\frac{\nu^a}{\nu^a + \delta_c^a} + B\nu^b + C\nu^c ,
\qquad \nu = \frac{\delta_c}{\sigma(M)} ,$$

evaluated at the growth-rescaled peak height
$\nu(M) / [D(z)/D(0)]$ (see {doc}`halo_model` for the normalisation trap).

Where it enters the observables:

- `b_sel_marg` weights the correlated-structure operators $I_1$, $J$ by
  $b(M,z)$ ({doc}`../systematics/bsel`);
- `bsel` builds the mass-averaged effective bias $b_{\rm eff}$ per bin
  ({doc}`../systematics/bsel`);
- `ShearPrjGl` (and the DES Y1 `shear_prj_frozen_physics`) multiplies the
  clustered channel by $b(M,z)\, b_{\rm sel}(\theta)\, \xi_{\rm NL}$
  ({doc}`../observables/shear_projection`);
- `Shear1h2hMax` scales the two-halo table, $b(M,z)\,\Delta\Sigma_{\rm hh}$
  ({doc}`../observables/second_halo_term`).

## The DataBlock contract

| DataBlock key | Meaning | Units / shape | Produced by | Consumed by |
|---|---|---|---|---|
| `haloModel/lnM` | mass axis | $\ln(M_\odot/h)$, `(100,)` | `halo_model` | `b_sel_marg`, `ShearPrjGl`, `bsel`, `Shear1h2hMax` |
| `haloModel/z` | redshift axis | `(50,)` | `halo_model` | same |
| `haloModel/bias` | Tinker-2010 bias $b(M,z)$, $\Delta = 200\bar\rho_m$ | dimensionless, `(50, 100)` | `halo_model` | same |

