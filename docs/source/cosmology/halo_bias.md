# Halo Bias

`Python` (producer) · `y3_cluster_cpp` (`y3_buzzard/`) · `Halo population`

$b(M, z)$ quantifies how strongly haloes cluster relative to the matter
field. In this pipeline it is **not a module of its own**: the
`halo_model` module publishes it as a DataBlock grid, and three consumers
interpolate it.

## Script

- Producer: [`y3_buzzard/halo_model_cosmosis.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/y3_buzzard/halo_model_cosmosis.py)
  via `y3_buzzard/haloModel/biasModel.py` and
  `cluster_toolkit.bias.bias_at_nu` — see {doc}`halo_model` for the module
  mechanics.
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
  $b(M,z)$ ({doc}`../selection/bsel`);
- `bsel` builds the mass-averaged effective bias $b_{\rm eff}$ per bin
  ({doc}`../selection/bsel`);
- `shear_prj_frozen_physics` multiplies the clustered channel by
  $b(M,z)\, b_{\rm sel}(\theta)\, \xi_{\rm NL}$
  ({doc}`../observables/shear_projection`).

## The DataBlock contract

| DataBlock key | Meaning | Units / shape | Produced by | Consumed by |
|---|---|---|---|---|
| `haloModel/lnM` | mass axis | $\ln(M_\odot/h)$, `(100,)` | `halo_model` | `b_sel_marg`, `shear_prj_frozen_physics`, `bsel` |
| `haloModel/z` | redshift axis | `(50,)` | `halo_model` | same |
| `haloModel/bias` | Tinker-2010 bias $b(M,z)$, $\Delta = 200\bar\rho_m$ | dimensionless, `(50, 100)` | `halo_model` | same |

