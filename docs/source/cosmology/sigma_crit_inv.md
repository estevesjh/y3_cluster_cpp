# Inverse Critical Surface Density

`Python` · `y3_cluster_cpp` · `Lensing geometry` · module `average_sigma_crit_inv` · `<1 ms/sample`

Computes the source-averaged inverse critical surface density
$\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)$ — the geometric factor that
converts $\Delta\Sigma$ into tangential shear. Both shear modules
(`Shear1hMisSel`, `shear_prj_frozen_physics`) multiply by it.

## Numerical framework

$$\langle \Sigma_{\rm crit}^{-1} \rangle(z_l)
= h_0 \int dz_s\; p(z_s + \delta_z)\,
  \frac{4\pi G}{c^2}\,
  \frac{D_A(z_l)\, \big[ D_A(z_s) - \frac{1+z_l}{1+z_s} D_A(z_l) \big]}{D_A(z_s)},$$

clipped at zero for sources in front of the lens. Per lens redshift:
1-D interpolation of $D_A$ from `distances`, source $p(z_s)$ shifted by
`delta_z`, trapezoidal integration over the source grid
($G = 4.517 \times 10^{-48}\,\mathrm{Mpc^3\,M_\odot^{-1}\,s^{-2}}$).

## Script

- Source: [`src/modules/average_sigma_crit_inv/average_sigma_crit_inv.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/src/modules/average_sigma_crit_inv/average_sigma_crit_inv.py)
  (`y3_cluster_cpp` @ `d7feb75`).
- Source $n(z)$ read at setup from
  `${Y3_CLUSTER_CPP_DIR}/data/test_cluster_Y1.fits` (HDU 6, columns
  `z_mid`, `bin1`).
- Loaded by CosmoSIS as a Python module.

## CosmoSIS setup

```ini
[average_sigma_crit_inv]
file = ${Y3_CLUSTER_CPP_DIR}/src/modules/average_sigma_crit_inv/average_sigma_crit_inv.py
z_min = 0.05
z_max = 0.80
z_bins = 50
```

- Requires `Y3_CLUSTER_CPP_DIR` (module path and the FITS $n(z)$).
- Ordering: after `cp_camb` (needs `distances/d_a`); before the shear
  modules.
- The reference run leaves `unity` at its default `F`, so the published
  average is the **physical** integral and the shear observable is
  $\gamma_t$. (Setting `unity = T` publishes
  $\langle\Sigma_{\rm crit}^{-1}\rangle \equiv 1$, turning the shear
  outputs into $\Delta\Sigma$ — used by the self-closure mock pipelines,
  see {doc}`../variants`.)

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `z_min`, `z_max` | lens-redshift grid range | — | 0.05, 0.80 |
| `z_bins` | number of lens-redshift points | — | 50 |
| `unity` | publish $\langle\Sigma_{\rm crit}^{-1}\rangle \equiv 1$ ($\Delta\Sigma$-observable mode) | — | F (default) |

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `cosmological_parameters/h0` | Hubble parameter | scalar | `consistency` |
| `photoz/delta_z` | source photo-$z$ shift nuisance | scalar | sampler (values file) |
| `distances/z`, `distances/d_a` | angular-diameter distance | Mpc, `(50,)` | `cp_camb` |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `average_sigma_crit_inv/zlense` | lens-redshift grid | `(50,)` | `Shear1hMisSel`, `shear_prj_frozen_physics` |
| `average_sigma_crit_inv/sci_average` | $\langle\Sigma_{\rm crit}^{-1}\rangle(z_l)$ | inverse surface density, in the convention matching the $10^{-12}$-scaled NFW $\Delta\Sigma$ tables ($M_\odot/(h\,\mathrm{pc}^2)$) so $\gamma_t$ is dimensionless; `(50,)` | same |

