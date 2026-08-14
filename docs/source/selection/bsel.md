# Selection Bias

Two modules run back-to-back to predict the scale-dependent optical
selection bias $b_{\rm sel}(\theta)$: the C++ `b_sel_marg` evaluates the
three $P[X]$ population operators, and the Python `bsel` closes them into
the two bias plateaus $(B_{\rm small}, B_{\rm large})$ that
`shear_prj_frozen_physics` consumes.

The model is
[Costanzi et al. 2026, PhRvD 113, 103508](https://ui.adsabs.harvard.edu/abs/2026PhRvD.113j3508C/abstract)
(arXiv:[2604.05833](https://arxiv.org/abs/2604.05833)); derivations in
{doc}`../math/index` (optical selection bias).

---

## The $\mathcal{P}[X]$ Operators

`C++` · `y3_cluster_cpp` · `Selection` · module `b_sel_marg` · `66 ms/sample`

Co-computes the three Costanzi-2026 scalars $(P_1, I_1, J)$ on the
12-bin $(z^{\rm ob}, \lambda^{\rm ob})$ wall in one fixed-GL pass.

### Script

- Model: [`src/models/p_operator_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/src/models/p_operator_t.hh)
  (`y3_cluster::P_operator`; `module_label()` hard-codes `b_sel_marg`, so
  **the ini section must keep this name**).
- Module driver: [`src/modules/b_sel_marg_cpu/BSelMargIntegrand.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/src/modules/b_sel_marg_cpu/BSelMargIntegrand.cc).
- Compiled library loaded by CosmoSIS:
  `${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/b_sel_marg_cpu/BSelMargIntegrand.so`.

### Numerical framework

$$\begin{aligned}
P_1 &= \int \mathcal{W}\, f_A\big(\theta, \theta_\lambda(\lambda^{\rm tr})\big), \qquad
I_1 = \int \mathcal{W}\, b(M,z)\,\xi_{\rm NL}(|\Delta\chi|, z^{\rm ob})\,
      \sigma(\theta)\, f_A, \\
J   &= \int \mathcal{W}\, b\,\xi_{\rm NL}\,\big(1 - \sigma(\theta)\big)\, f_A,
\end{aligned}$$

over $(z, \ln M, \lambda^{\rm tr}, \theta)$. In the paper's notation
these are specialisations of the projection-average operator
$\mathcal{P}[X]$: $P_1 = \mathcal{P}[1]$,
$I_1 = \mathcal{P}[b\,\xi_{\rm NL}\,\sigma(\theta)]$, and
$J = I_2 - I_1$ with $I_2 = \mathcal{P}[b\,\xi_{\rm NL}]$. The
population weight is
$\mathcal{W} \propto (dV/d\Omega\,dz)\, (dn/d\ln M)\,
P_{\rm HOD}(\lambda^{\rm tr}|M,z)\,\lambda^{\rm tr}\, 2\pi\sin\theta$,
line-of-sight exclusion $\theta > \theta_{\rm excl}(z)$, and sigmoid
$\sigma(\theta) = [1 + e^{-k(\theta - \theta_0)}]^{-1}$,
$k = 2.5/\theta_\lambda$, $\theta_0 = \theta_\lambda/2$. The photo-$z$
width $\sigma_z(z)$ comes from the compiled-in table
`src/models/z_kernel_data.hh`, and $\Omega(z)$ is deliberately absent
({doc}`../modules/survey_area`). Fixed GL
throughout — ring + foreground/background $\log|\Delta\chi|$ wings in $z$,
one cached $\theta$ grid split at $\theta_\lambda$, mass integral
pre-contracted — $\sim 74$ ms for all 12 bins ($\sim 10^3\times$ faster
than the PAGANI reference benchmarks, {doc}`../variants`). Agreement with
the Python reference `sel_bias._P_operator`: $\leq 1\%$.

---

### CosmoSIS setup

```ini
[b_sel_marg]
file = ${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/b_sel_marg_cpu/BSelMargIntegrand.so
lnm_low  = 29.9336
lnm_high = 35.6814
n_lt     = 60
n_lnm    = 24
n_theta  = 10
n_zring  = 20
n_zouter = 20
zo_low     = 0.20 0.20 0.20 0.20  0.35 0.35 0.35 0.35  0.50 0.50 0.50 0.50
zo_high    = 0.35 0.35 0.35 0.35  0.50 0.50 0.50 0.50  0.65 0.65 0.65 0.65
lambda_bin = 0 1 2 3  0 1 2 3  0 1 2 3
```

- Ordering: after `halo_model` (bias, $\xi_{\rm NL}$), `MfTinker`,
  `cp_camb`; before `bsel`.

### Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `zo_low`, `zo_high`, `lambda_bin` | 12-entry bin wall; $z^{\rm ob}$ = bin midpoint, richness centre from `lob_center(bin)` | — | wall above |
| `lnm_low`, `lnm_high` | mass GL limits | $\ln(M_\odot/h)$ | 29.9336, 35.6814 |
| `n_lt` | GL nodes in $\lambda^{\rm tr}$ on $(0, \lambda^{\rm ob}_{\rm centre}]$ | — | 60 |
| `n_lnm` | GL nodes in $\ln M$ | — | 24 |
| `n_theta` | angular GL nodes, split at $\theta_\lambda$ | — | 10 |
| `n_zring`, `n_zouter` | redshift nodes: ring band / each fg-bg wing | — | 20, 20 |

### DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `distances/{z, d_c}` | comoving distance ($\times h_0$ → cMpc/$h$) | Mpc | `cp_camb` |
| `distances/{z, d_a}` | volume element via `DV_DO_DZ_t` | Mpc | `cp_camb` |
| `mass_function/*` | HMF via `HMF_t` | — | `MfTinker` |
| `haloModel/{lnM, z, bias}` | halo bias grid | `(50, 100)` | `halo_model` |
| `xi_nl/{r, z, xi_nl}` | nonlinear correlation function | `(50, 128)` | `halo_model` |
| `cluster_mor/*` | shifted-Poisson HOD parameters via `MOR_HOD_t` | — | sampler (values file) |
| `cosmological_parameters/{omega_m, omega_nu, h0}` | mass-axis shift, unit conversions | scalars | `consistency` |

### DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `b_sel_marg_P1/vals` | $P_1$ (projection normalisation) | `(12,)` | `bsel` |
| `b_sel_marg_I1/vals` | $I_1$ (bias-weighted, sigmoid-on) | `(12,)` | `bsel` |
| `b_sel_marg_J/vals` | $J = I_2 - I_1$, computed directly — the difference cancels catastrophically at large $\theta$ where $\sigma(\theta) \to 1$ | `(12,)` | `bsel` |

## Analytic Closure for $b_{\rm sel}(\theta)$

`Python` · `y3_cluster_cpp` (`y3_buzzard/`) · `Selection` · module `bsel` · `16 ms/sample`

Closes the $(P_1, I_1, J)$ operators into the per-bin bias plateaus and
publishes the two scalars from which the full $\theta$ dependence is
analytic.

### Script

- Source: [`y3_buzzard/bsel.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/y3_buzzard/bsel.py)
  (`y3_cluster_cpp` @ `d7feb75`).
- EMG projection-kernel coefficients imported from
  `y3_buzzard/prj_params.py` (`PrjParams.default()`), not from the
  DataBlock.

### Numerical framework

Since only the sigmoid depends on $\theta$, the $\lambda^{\rm tr}$
marginalisation factorises exactly into two scalars:

$$\langle b_{\rm sel}\rangle(\theta) = B_{\rm small} +
(B_{\rm large} - B_{\rm small})\,\sigma(\theta),$$

with, per latent richness,
$b_\infty = b_{\rm eff}(1 + 0.13\,\delta_{\rm prj})$ and
$b_{\rm zero} = [(\lambda^{\rm ob} - \lambda^{\rm tr}) - P_1 - b_\infty I_1]/J$.
$J$ is used directly as the denominator (never $I_2 - I_1$). The
marginalisation weight combines the GL weights, the EMG
$P(\lambda^{\rm ob}|\lambda^{\rm tr}, z)$, and the mass-integrated HOD.
Two guarded historical bugs: the $h_0$ factor on $\chi$ (12% sigmoid
shift) and the $dn/dM$ vs $dn/d\ln M$ convention (halved $B_{\rm small}$).
Full model: {doc}`../math/index`.

### CosmoSIS setup

```ini
[bsel]
file = ${Y3_CLUSTER_CPP_DIR}/y3_buzzard/bsel.py
lob      = 25.0 37.5 52.5 130.0
zob      = 0.275 0.425 0.575
n_theta  = 32
theta_lo = 1e-4
theta_hi = 5e-3
n_ltr          = 128
ltr_lo         = 1.0
ltr_hi_factor  = 3.0
```

- Ordering: immediately after `b_sel_marg`; before
  `shear_prj_frozen_physics`.

### Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `lob`, `zob` | richness / photo-$z$ bin centres | — | 4 + 3 values |
| `n_theta`, `theta_lo`, `theta_hi` | legacy tabulated $\theta$ grid (geometric) | rad | 32, $10^{-4}$, $5\times10^{-3}$ |
| `n_ltr`, `ltr_lo`, `ltr_hi_factor` | $\lambda^{\rm tr}$ GL marginalisation on $[1,\, 3\lambda^{\rm ob}]$ | — | 128, 1.0, 3.0 |

### DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `b_sel_marg_{P1, I1, J}/vals` | the $P[X]$ operators | `(12,)` each | `b_sel_marg` |
| `mass_function/{m_h, z, dndlnmh}` | HMF (mass axis rescaled by $\Omega_m - \Omega_\nu$ explicitly) | — | `MfTinker` |
| `haloModel/{m_h, z, bias}` | halo bias for $b_{\rm eff}$ | `(50, 100)` | `halo_model` |
| `cluster_mor/*` | HOD parameters | — | sampler (values file) |
| `distances/{z, d_c}`, `cosmological_parameters/h0` | $\chi(z^{\rm ob})$ in cMpc/$h$ for $\theta_\lambda$ | — | `cp_camb`, `consistency` |

### DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `b_sel_marginalised/b_small` | small-scale plateau $B_{\rm small} = \langle b_{\rm zero}\rangle_{\lambda^{\rm tr}}$ | `(3, 4)` | `shear_prj_frozen_physics` |
| `b_sel_marginalised/b_large` | large-scale plateau $B_{\rm large} = \langle b_\infty\rangle_{\lambda^{\rm tr}}$ | `(3, 4)` | `shear_prj_frozen_physics` |
| `b_sel_marginalised/b_eff` | mass-averaged halo-bias aggregate per bin — the paper's $b_{\rm halo}$ (unselected bias) | `(3, 4)` | diagnostics |
| `b_sel_marginalised/{lob, zob, theta, vals}` | backward-compatible $b_{\rm sel}(\theta)$ tabulation | `(4,)`, `(3,)`, `(32,)`, `(4, 3, 32)` | legacy consumers |

