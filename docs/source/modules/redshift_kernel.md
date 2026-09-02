# Redshift Kernel

`C++ / Python` (model, not a CosmoSIS module) · `y3_cluster_cpp` · `Selection`

The photometric-redshift kernel $P(z^{\rm ob}\mid z^{\rm tr})$ relates a
cluster's true and observed redshift. It enters the counts through the
observed-redshift kernel $\mathcal S_j$, and the selection-bias and
projection operators through the richness-dependent scatter
$\sigma_z(z)$ and the line-of-sight weight $w_z$.

## Script

- Compiled scatter table: [`src/models/z_kernel_data.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/models/z_kernel_data.hh)
  — `z_kernel_z()` / `z_kernel_sigma()`, the calibrated $\sigma_z(z)$
  arrays baked into the binary (**not** read from the DataBlock).
- Bin kernel (C++): `richness_zkernel` in
  [`src/models/richness_kernel_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/models/richness_kernel_t.hh)
  — the Gaussian CDF difference $\mathcal S_j$.
- Python mirror: `_K_j` in
  [`src/modules/sel_function/sel_function.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/sel_function/sel_function.py)
  (uses the per-bin ini `sigma_z`).
- Line-of-sight weight: `build_z_grid_` in
  [`src/models/sigma_prj_frozen_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/models/sigma_prj_frozen_t.hh)
  / `sigma_prj_t.hh` and the $z$-grid builder of
  [`src/models/p_operator_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/models/p_operator_t.hh).

## Numerical framework

The full integral: the observed-redshift kernel is the bin integral of a
Gaussian $P(z^{\rm ob}\mid z^{\rm tr})$ of width $\sigma_z$,

$$\mathcal S_j(z^{\rm tr}) =
\int_{\Delta z_j}\! dz^{\rm ob}\;
P(z^{\rm ob}\mid z^{\rm tr}, \Delta\lambda_i)
= \Phi\!\left(\frac{z_j^{\max} - z^{\rm tr}}{\sigma_z}\right)
- \Phi\!\left(\frac{z_j^{\min} - z^{\rm tr}}{\sigma_z}\right),$$

closed-form — no numerical integration anywhere in this kernel. Two
scatter sources are in use:

- **Abundance side** (`sel_function`): the per-bin ini value
  `sigma_z = 0.03`, constant in redshift.
- **Selection-bias / projection side** (`b_sel_marg`,
  `shear_prj_frozen_physics`): the compiled $\sigma_z(z)$ table of
  `z_kernel_data.hh`, evaluated at the ring endpoints by bisection of
  $z \pm \sigma_z(z) = z^{\rm ob}$ ({doc}`../numerics/index`, the
  shear-projection recipe, Step 4). Along the line of sight these
  operators use the compact parabolic weight
  $w_z(z, z^{\rm ob}) = \max\!\big(0,\, 1 - u^2\big)$,
  $u = (z - z^{\rm ob})/\sigma_z(z)$, instead of a Gaussian — same
  width, finite support.

## Consumed by

| Consumer | What it uses | Where |
|---|---|---|
| `sel_function` | $\mathcal S_j$ with ini `sigma_z` | the $S_{ij}$ tensor ({doc}`../systematics/sel_function`) |
| `NumCountsSel`, `Shear1hMisSel` | $\mathcal S_j$ folded inside `S_stack` | {doc}`../observables/number_counts`, {doc}`../observables/shear_halo` |
| `b_sel_marg` | $\sigma_z(z)$ table for the ring band | {doc}`../systematics/bsel` |
| `shear_prj_frozen_physics` | $\sigma_z(z)$ table + parabolic $w_z$ | {doc}`../observables/shear_projection` |

```{note}
Because the C++ side reads the compiled table rather than the DataBlock,
changing the photo-$z$ scatter model requires a recompile (or the
`sel_function` ini for the abundance side). `photoz/delta_z` — the
*source*-population shift used by `average_sigma_crit_inv` — is a
different quantity ({doc}`../cosmology/sigma_crit_inv`).
```
