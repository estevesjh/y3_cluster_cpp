# Mathematical Framework

The mathematical definition of each pipeline observable and of the
selection functions that enter them, in the nomenclature of the
optical-selection paper. Definitions are implementation-independent; the
numerical treatment lives in {doc}`../numerics/index` and the code
mapping in the per-module pages linked from {doc}`../running`.

The main published reference for the forward model — cluster number
counts and population-averaged lensing with miscentering, and the
CosmoSIS software framework implementing them — is
[DES Cluster et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230906593A/abstract)
(arXiv:[2309.06593](https://arxiv.org/abs/2309.06593)). The optical
selection-bias and projection-lensing model ("Costanzi-2026" throughout
these chapters) is
[Costanzi et al. 2026, PhRvD 113, 103508](https://ui.adsabs.harvard.edu/abs/2026PhRvD.113j3508C/abstract)
(arXiv:[2604.05833](https://arxiv.org/abs/2604.05833)).

Further source material: [`RichnessSelection/docs/richness_selection_function.tex`](https://github.com/estevesjh/RichnessSelection/blob/main/docs/richness_selection_function.tex)
(selection functions and richness–mass models),
[`RichnessSelection/docs/richness_selection.tex`](https://github.com/estevesjh/RichnessSelection/blob/main/docs/richness_selection.tex) and
[`delta_sigma_prj_derivation.tex`](https://github.com/estevesjh/RichnessSelection/blob/main/docs/delta_sigma_prj_derivation.tex) (projection lensing),
[`docs/pipeline_modules.tex`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/docs/pipeline_modules.tex) and [`docs/projection_lensing_paper.tex`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/docs/projection_lensing_paper.tex) in this
repository.

## Number counts

The number-counts observable and every population-weighted quantity in the
pipeline are built from a single operator. `NumCountsSel`, `Shear1hMisSel`,
`MassWeightedSel`, and `BiasWeightedSel` all share the same C++ template
`NOperatorSelRadial<F>` / `NOperatorSelScalar<F>`
(`src/models/n_operator_sel_t.hh`); each instantiation plugs a different
integrand weight $f$ into the generic operator

$$
N_i[f](R) \;=\; \int d\ln M \int dz\;
  \Omega(z)\,\frac{dV}{d\Omega\,dz}\,n(M, z)\,
  S_{ij}(\ln M, z)\,f(R, \ln M, z),
$$

where $\Omega(z)$ is the survey-area factor, $dV/d\Omega\,dz$ the comoving
volume element, $n(M,z) = dn/d\ln M$ the halo mass function, and
$S_{ij}(\ln M, z)$ the richness-selection tensor published by
`sel_function`,

$$
S_{ij}(\ln M, z) = S_i(\ln M, z)\cdot \mathcal S_j(z)
 = \Big[\int d\lambda^{\rm tr}\, \mathcal S_i(\lambda^{\rm tr}, z)\,
   P_{\rm HOD}(\lambda^{\rm tr} \,|\, M, z)\Big]\cdot \mathcal S_j(z),
$$

with $\mathcal S_i$ the observed-richness kernel (a closed-form CDF
difference of the Costanzi projection kernel
$P(\lambda^{\rm ob}|\lambda^{\rm tr}, z)$ — see
[Selection functions](#selection-functions)) and $\mathcal S_j$ the
observed-redshift kernel (a Gaussian CDF difference). The weight $f$ selects the observable:

| Weight $f$ | Module | Observable built |
|---|---|---|
| $1$ | `NumCountsSel` | $N_i$ cluster number count |
| $M$ | `MassWeightedSel` | $\langle M\rangle_i = N_i[M]/N_i$ |
| $b(M, z)$ | `BiasWeightedSel` | $\langle b\rangle_i = N_i[b]/N_i$ |
| $\gamma_t^{1h,\rm full}(R; M, z)$ | `Shear1hMisSel` | $N_i[\gamma_t^{1h,\rm full}](R)$, centred + miscentered |

The cluster number counts are the $f = 1$ instantiation, $N_i[1]$,
published as `numcountssel/vals` — 12 bins in production (4 richness bins
$\times$ 3 photo-$z$ bins). The likelihood compares these 12 values
directly against the mock data vector.

**Production binning and grids.** DES Y3 richness bin edges are
$[20, 30, 45, 60, 200]$ with arithmetic centres $\{25, 37.5, 52.5, 130\}$
(the centre $130$ replaces a legacy value of $100$; the C++ evaluators
hard-code the correct centres — `sigma_prj_t.hh` `default_lob_centers()`,
`p_operator_cuhre_t.hh` `lob_center()` — or read a `lob_centers` ini
override). Photo-$z$ bins are
$[0.20, 0.35] \cup [0.35, 0.50] \cup [0.50, 0.65]$ with midpoints
$\{0.275, 0.425, 0.575\}$. The $(\ln M, z_{\rm true})$ integration runs
over the shared grid on which `sel_function` tabulates $S_{ij}$ and serves
it via `Interp2D`; the production grid sizes are `n_lnm = 192` (the
whole-pipeline optimum — coarser grids force the downstream adaptive
integrator to refine more and cost back the savings) and
`n_z_shared = 64`. Evaluators read flat 1-D wall axes (`zo_low`,
`zo_high`, `lambda_bin`, optionally `radii`) of length
$N_{\rm grid} = N_{z^{\rm ob}}\cdot N_{\lambda^{\rm ob}}\cdot N_R$; the
smoke setups use $N_{\rm grid}=12$ for scalar observables and $120$ for
radial ones ($N_R = 10$).

*Source: [`docs/pipeline_modules.tex`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/docs/pipeline_modules.tex) §Observables and the shear
composition / The $N_i[f]$ operator.*

## Cluster lensing

The lensing prediction separates physically distinct contributions — the
(mis)centered one-halo profile of the selected cluster, the
projection-selected line-of-sight contribution, and, in the fiducial
composition, the standard two-halo term. This chapter defines each and how
they combine into the theory vector.

### One-halo lensing and miscentering

The full DES Y3 cluster-lensing model does not assume the identified
central galaxy coincides with the dark-matter halo centre. A fraction
$f_{\rm mis}$ of redMaPPer clusters are miscentred: their identified centre
is offset from the true halo centre by a 2-D radial distance $R_{\rm mis}$
drawn from a kernel $P(R_{\rm mis}\,|\,\tau_{\rm mis}, R_\lambda)$. For a
cluster of mass $M$ at lens redshift $z$ the modelled profile is a
two-component mixture:

$$
\Delta\Sigma_{\rm cl}(R \,|\, M, z, \theta_{\rm mis})
 = (1 - f_{\rm mis})\,\Delta\Sigma_{\rm NFW}(R, M)
 + f_{\rm mis}\,\Delta\Sigma_{\rm mis}(R, M; \tau_{\rm mis}),
$$

where $\theta_{\rm mis} = \{f_{\rm mis}, \tau_{\rm mis}\}$ are the
miscentering nuisance parameters. $\Delta\Sigma_{\rm NFW}$ is the analytic
Wright & Brainerd (2000) projected differential surface density at the halo
centre (the centred NFW spline `haloModel/dSigma_nfw` is built by
`halo_model` with the Child-18 concentration; the miscentred lookup tables
described below use $c = 4$). $\Delta\Sigma_{\rm mis}$ is the same NFW
profile averaged over an azimuthal-and-radial offset distribution:

$$
\Delta\Sigma_{\rm mis}(R, M; \tau_{\rm mis})
 = \int_0^\infty dR_{\rm mis}\,
   P(R_{\rm mis}\,|\,\tau_{\rm mis}, R_\lambda(M))\,
   \widetilde{\Delta\Sigma}_{\rm NFW}(R \,|\, R_{\rm mis}, M),
$$

$$
\widetilde{\Delta\Sigma}_{\rm NFW}(R \,|\, R_{\rm mis}, M)
 = \frac{1}{2\pi}\int_0^{2\pi} d\varphi\;
   \Delta\Sigma_{\rm NFW}\!\left(\sqrt{R^2 + R_{\rm mis}^2 + 2RR_{\rm mis}\cos\varphi},\, M\right).
$$

Two offset kernels are supported (selected by the `kernel` string in
`src/models/nfw_dsigma_mis.hh`):

- **single** — a delta function at fixed offset,
  $P_{\rm single}(R_{\rm mis}\,|\,\tau_{\rm mis}, R_\lambda) =
  \delta_D(R_{\rm mis} - \tau_{\rm mis} R_\lambda)$. Used by the
  projection branch (`shear_prj`), because it factors cleanly into the
  $\theta = R_{\rm mis}/D_A(z^{\rm ob})$ angular integral.
- **gamma** — the DES Y3 redMaPPer miscentring calibration
  ([Kelly et al. 2024, MNRAS 533, 572](https://ui.adsabs.harvard.edu/abs/2024MNRAS.533..572K/abstract),
  arXiv:[2310.13207](https://arxiv.org/abs/2310.13207) — the updated
  analysis of
  [Zhang et al. 2019, MNRAS 487, 2578](https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.2578Z/abstract))
  as used by Costanzi-2026,

  $$
  P_\gamma(R_{\rm mis}\,|\,\tau_{\rm mis}, R_\lambda)
   = \frac{R_{\rm mis}}{(\tau_{\rm mis} R_\lambda)^2}\,
     \exp\!\left[-\frac{R_{\rm mis}}{\tau_{\rm mis} R_\lambda}\right],
  $$

  a Rayleigh-shaped kernel calibrated against DES Y3 redMaPPer centring on
  Buzzard mocks. This is the kernel used for the lensing branch — the
  convolution entering the `Shear1hMisSel` weight in the production
  pipeline.

The DES Y3 fiducial values, calibrated jointly against X-ray and SZ-derived
true centres, are $f_{\rm mis} \simeq 0.22 \pm 0.06$ and
$\tau_{\rm mis} \simeq 0.17 \pm 0.04$ (in units of $R_\lambda$), with
$R_\lambda(\lambda) = (\lambda/100)^{0.2}\,h^{-1}\,\mathrm{Mpc}$.

**Pre-computed convolution tables.** The convolution above is non-trivially
expensive: one azimuthal integral per $(R, M, R_{\rm mis})$ plus an outer
$R_{\rm mis}$ integral against $P(R_{\rm mis})$. Inlining it in the inner
integration loop of `Shear1hMisSel` (called at every $(R, \ln M, z)$ node,
$\sim 10^5$ nodes/sample) is prohibitive. Instead the module reads the same
lookup tables already used by the projection branch, under
`data/nfw_off_center/` (files `table_1000_1e-03_5e+03_<kernel>_logx.txt`,
`..._logxmis.txt`, `..._log_deltasigma_<kernel>.txt`,
`..._log_sigma_<kernel>.txt` with `<kernel>` $\in$ {single, gamma}). Each
is a $1000\times1000$ log-log grid in
$(x = R/r_s,\; x_{\rm mis} = R_{\rm mis}/r_s)$ returning
$\log[\Delta\Sigma_{\rm mis}/(2 r_s \delta_c \rho_{\rm crit})]$. At
evaluation time the cluster's $r_s = r_{200}/c$ is computed from $M$, $c$,
$\rho_{\rm crit}$ and the stored profile is rescaled — no on-the-fly
convolution is performed. The tables are keyed on
$(\ln(R/r_s), \ln(R_{\rm mis}/r_s))$ with $r_s$ derived from
$\ln M_{\rm phys}$ and $c = 4$.

**Shear1hMisSel is $N_i$-weighted; divide by NumCountsSel.**
`Shear1hMisSel` is *not* the per-cluster 1-halo shear
$\gamma_t^{1h,\rm full}(R)$; it is the integral of the 1-halo shear over
the cluster population in the bin, weighted by
$\Omega(z)\,dV/d\Omega\,dz\;n(M,z)\,S_{ij}$. Inside the operator template
the weight is

$$
\gamma_t^{1h,\rm full}(R; M, z) =
 \Big[(1 - f_{\rm mis})\,\Delta\Sigma_{\rm NFW}(R, M)
 + f_{\rm mis}\,\Delta\Sigma_{\rm mis}(R, M; \tau_{\rm mis} R_\lambda)\Big]\,
 \Sigma_{\rm crit}^{-1}(z),
$$

built from the centred NFW spline `haloModel/dSigma_nfw` plus the
gamma-kernel miscentered table (loaded once at module construction;
$f_{\rm mis}$, $\tau_{\rm mis}$ refreshed each sample from the
`miscentering` datablock section, with in-code fallbacks $0.22$, $0.17$;
the per-bin $R_\lambda$ set via a `set_bin` hook forwarded from the
wall-grid cursor). The per-cluster average is recovered by dividing by the
number-count integral at the same wall point:

$$
\langle\gamma_t^{1h,\rm full}\rangle_i(R)
 = \frac{N_i[\gamma_t^{1h,\rm full}](R)}{N_i[1]}
 = \frac{\mathtt{shear1hmissel/vals}}{\mathtt{numcountssel/vals}}.
$$

This division lives in the likelihood (`y3_buzzard/likelihood_cp.py`), not
inside a single module. Because $f_{\rm mis}$ and $\tau_{\rm mis}$ are
scalars (no $M$ or $z$ dependence in the fiducial parameterisation), the
integral remains linear in the two pieces,

$$
N_i[\gamma_t^{1h,\rm full}](R)
 = (1 - f_{\rm mis})\, N_i[\gamma_t^{1h,\rm cen}](R)
 + f_{\rm mis}\, N_i[\gamma_t^{1h,\rm mis}](R),
$$

so a chain varying only $(f_{\rm mis}, \tau_{\rm mis})$ at fixed cosmology
could in principle cache the two $N_i$ vectors and re-mix per MCMC step.
Setting `miscentering/f_mis = 0` recovers bit-identical closure with the
centred branch. At the fiducial parameters the small-$R$
$\langle\gamma_t^{1h}\rangle$ is suppressed by $\sim 30\%$ at
$R \lesssim 0.3\,h^{-1}\,\mathrm{Mpc}$ relative to the centred profile.

*Source: [`docs/pipeline_modules.tex`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/docs/pipeline_modules.tex) §Miscentering selection on
$\Delta\Sigma$.*

### Projection lensing: $\Sigma_{\rm prj}$ and $\Delta\Sigma_{\rm prj}$

#### The full two-halo model

The master equation (Costanzi 2026 Eq. 13) for the two-halo projected
surface density around a richness-selected cluster is

$$
\begin{aligned}
\langle\Sigma^{\rm prj}(R \mid \lambda^{\rm ob}, z^{\rm ob})\rangle
&= 2\pi \int d\theta\,\sin\theta
  \int dz\,\frac{dV}{dz\,d\Omega}(z)\, w_z(z, z^{\rm ob})
  \int dM\, n(M, z) \\
&\quad\times
  \bigl[\, 1 + b(M, z)\, b_{\rm sel}(\theta;\lambda^{\rm ob},z^{\rm ob})\,
  \xi_{\rm NL}(|\Delta r|, z^{\rm ob}) \,\bigr]\,
  \Sigma_{\rm mis}\bigl(R \mid M, z, R_{\rm mis}=\theta\,D_A(z^{\rm ob})\bigr),
\end{aligned}
$$

where the 3-D comoving separation is the **exact chord**
$|\Delta r|^2 = \chi(z)^2 + \chi(z^{\rm ob})^2 -
2\,\chi(z)\,\chi(z^{\rm ob})\cos\theta$ (the $\Delta\chi$-only
approximation errs by 35% at $\theta=0.1\,\theta_\lambda$ and $>1000\%$ at
$2\,\theta_\lambda$, because near the ring the transverse term dominates),
and exclusion is a **line-of-sight slab**, $\xi_{\rm NL}\to 0$ for
$\theta\le\theta_{\rm excl}(z)$ with
$\cos\theta_{\rm excl}(z) =
[\chi(z)^2+\chi(z^{\rm ob})^2-R_{\rm excl}^2]/[2\chi(z)\chi(z^{\rm ob})]$ —
not a 3-D ball mask on $|\Delta r|$. The kernel $\Sigma_{\rm mis}$ is the
azimuth-averaged miscentered NFW surface density of the neighbour,

$$
\Sigma_{\rm mis}(R'\mid M,z,R_{\rm mis})
= \int_0^{2\pi}\!d\varphi\;
  \Sigma_{\rm NFW}\!\Bigl(\sqrt{R'^{\,2}+R_{\rm mis}^2-2R'R_{\rm mis}\cos\varphi}\,\Big|\,M,z\Bigr).
$$

This is the cylindrical specialisation of the Cooray–Sheth two-halo term
(Eqs. 86–87 of the halo-model review): starting from $\xi_{2h}$ built from
halo profiles $u(\cdot\mid M)$ around the halo-halo correlator
$\xi_{hh}$, fix the cluster at the origin, drop its own profile (the
1-halo term is separate), and change the neighbour's coordinates
$(\mathbf{R}_\perp,\chi_\parallel)\to(\theta,z)$ with volume element
$d^2R_\perp\,d\chi_\parallel = 2\pi\sin\theta\,d\theta\cdot
(dV/dz\,d\Omega)\,dz$; the azimuth average around the neighbour's offset
$R_{\rm mis}=\theta\,\chi(z_{\rm cls})$ produces $\Sigma_{\rm mis}$.
Linear deterministic bias,
$\xi_{hh}(r\mid M_{\rm cls},M)\approx b_{\rm cls}\,b(M,z)\,\xi_{\rm lin}(r)$,
is then upgraded in three steps for a richness-selected target: (i)
$\xi_{\rm lin}\to\xi_{\rm NL}$ (halofit), because the 1h–2h transition at
$\sim R_{\rm excl}$ is nonlinear; (ii) the LoS-slab exclusion above; (iii)
$b_{\rm cls}\to b_{\rm sel}(\theta;\lambda^{\rm ob},z^{\rm ob})$ (see
{doc}`../math/index`), plus the uncorrelated cosmological
mean as the $+1$ inside the bracket.

#### The channel split: $\Sigma_{\rm rnd}$ vs $\Sigma_{\rm cl+LSS}$

The `1` term integrates to the *mean cosmological* projected surface
density in the photo-$z$ window, $\Sigma_{\rm rnd}(R)$ — spatially
near-uniform; the $b\,b_{\rm sel}\,\xi_{\rm NL}$ term is the
*correlation-excess* two-halo contribution $\Sigma_{\rm cl+LSS}(R)$. In
$\Delta\Sigma = \bar\Sigma(<R) - \Sigma(R)$ a uniform field satisfies
$\bar\Sigma(<R)\equiv\Sigma_{\rm rnd}$, so $\Delta\Sigma_{\rm rnd}(R)=0$:
the mean photo-$z$-window surface density cancels from the measured excess
(the classical Sheldon 2009 / Zu 2014 / Melchior 2017
random-point-subtraction statement). Numerically, with $\theta$ truncated
at $\theta_{\max}=R_{\max}/\chi(z_{\rm cls})$, $\Delta\Sigma_{\rm rnd}$ is
not exactly zero but a truncation-dependent boundary term (the rnd piece
grows $\sim 15\%$ going $R_{\max}=30\to 60\,h^{-1}\mathrm{Mpc}$ while
cl+LSS changes $<1\%$). The pipeline therefore returns the cl+LSS piece by
default for both observables.

#### $\Delta\Sigma_{\rm prj}$ and the integration limit

The lensing observable is the excess surface density,

$$
\langle\Delta\Sigma^{\rm prj}(R')\rangle \equiv
\bar\Sigma^{\rm prj}(<R') - \langle\Sigma^{\rm prj}(R')\rangle,
\qquad
\bar\Sigma^{\rm prj}(<R') \equiv
\frac{2}{R'^{\,2}}\int_0^{R'}\! s\,\langle\Sigma^{\rm prj}(s)\rangle\,ds .
$$

The excess functional acts only on the radial argument and therefore
commutes with the outer $(\theta, z, M)$ integrals: the
$\Delta\Sigma^{\rm prj}$ prediction is the $\Sigma^{\rm prj}$ machinery
with the kernel swap
$\Sigma_{\rm mis} \to \Delta\Sigma_{\rm mis} \equiv
\bar\Sigma_{\rm mis}(<R') - \Sigma_{\rm mis}(R')$ — a different lookup
against the same offset-NFW tables. Because
$\Delta\Sigma_{\rm mis}(R' \mid R_{\rm mis})$ has compact support around
$R_{\rm mis} \sim R'$, the line-of-sight truncation is adaptive,

$$
\theta_{\max}^{\Delta\Sigma}
= \frac{C\,R'_{\max}}{\chi(z_{\rm cls})},
\qquad C\simeq 3
\ \ (\text{not below} \sim 2),
$$

replacing the fixed $R_{\max} = 30\,h^{-1}$Mpc cut. Full derivation:
[`delta_sigma_prj_derivation.tex`](https://github.com/estevesjh/RichnessSelection/blob/main/docs/delta_sigma_prj_derivation.tex) in `RichnessSelection`.

### The shear composition

The pipeline emits two observables for the likelihood: the 12
cluster-count bins $N_i[1]$, and the summed tangential shear (length 120,
i.e. $12 \times 10\,R$ points):

$$
\gamma_t^{\rm theory}(R \,|\, i, j)
 = \langle\gamma_t^{1h}\rangle_i(R) + \gamma_t^{\rm prj}(R \,|\, \lambda^{\rm ob}, z^{\rm ob})
 = \frac{\mathtt{shear1hsel/vals}}{\mathtt{numcountssel/vals}} + \mathtt{shear\_prj/vals}.
$$

In the language of the Costanzi-2026 paper:
$\langle\gamma_t^{1h}\rangle_i(R)$ is the one-halo lensing signal expected
for a perfectly-centred cluster population, and $\gamma_t^{\rm prj}(R)$ is
the projection-effect correction — the additional shear contributed by
miscentering plus correlated large-scale structure, integrated over the
same cluster sample via the photo-$z$ kernel $w_z$.

**Why the sum is linear and exact.** Both modules emit the tangential
shear $\gamma_t(R) = \Delta\Sigma(R)\,\Sigma_{\rm crit}^{-1}(z)$, *not*
the reduced shear $g_t = \gamma_t/(1 - \Sigma\,\Sigma_{\rm crit}^{-1})$.
The earlier reduced-shear form coupled the two pieces through its
$1/(1-x)$ denominator, making the observable nonlinear in the
$\Delta\Sigma$ decomposition and blocking the additive split. Dropping the
denominator (retired 2026-05-11) recovers the linear-in-$\Delta\Sigma$
form, so the sum is exact, not a leading-order approximation.
`likelihood_cp.py` assembles the summed theory vector internally and
compares it against the single `data_Shear` entry of the mock npz.

**Why `shear_prj` is not divided by `NumCountsSel`.** `shear_prj` does
*not* pass through the $N_i[f]$ template — it produces a different
integrand (the master equation above), with three structural differences
from the $N_i[f]$ operator:

- **No $S_{ij}$.** A parabolic photo-$z$ kernel
  $w_z(z, z^{\rm ob}) = \max(0, 1 - u^2)$ replaces the richness-selection
  tensor; the module evaluates per $(\lambda^{\rm ob}, z^{\rm ob})$ wall
  point, i.e. it is already per richness bin and needs no population
  normalisation.
- **No $\Omega(z)$.** The survey-area factor cancels between
  $\Sigma^{\rm prj}$ and its normalisation, so it is not applied.
- **Off-centred NFW $\Sigma_{\rm mis}$ with the single kernel.** The
  1-halo density uses a single-offset miscentered NFW (offset
  $R_{\rm mis} = \theta D_A(z^{\rm ob})$), not the centred
  $\Sigma_{\rm NFW}$ nor the gamma-kernel mixture used by
  `Shear1hMisSel`. Here the offset *is* the $\theta$-integration
  variable, so a $\delta$-kernel in $R_{\rm mis}$ is the right physics
  and a gamma kernel would double-integrate over $R_{\rm mis}$.

The two branches therefore integrate different kernels (centred NFW vs
miscentered $\Sigma_{\rm mis}$) against different weights ($S_{ij}$ vs
$w_z$): only the $N_i$-weighted 1-halo branch needs the $N_i[1]$ division,
while $\gamma_t^{\rm prj}$ enters the sum directly. Both the $[1]$ term
(`rnd`) and the $[b\,b_{\rm sel}\,\xi_{\rm NL}]$ term (`cl`) are
accumulated separately, and the same kernel construction produces
$\Delta\Sigma^{\rm prj}$ and
$\gamma_t^{\rm prj} = \Delta\Sigma^{\rm prj}\,\Sigma_{\rm crit}^{-1}$ in
one pass; `sigma_prj/vals`, `dsigma_prj/vals`, and the `rnd`/`cl`
subfields are published for diagnostics.

*Source: [`docs/pipeline_modules.tex`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/docs/pipeline_modules.tex) §Observables and the shear
composition.*

### The standard two-halo term (fiducial 1h+2h)

The `halo_model` module (`y3_buzzard/halo_model_cosmosis.py`) is the
single most configurable module in the pipeline: it publishes the
`haloModel/bias` and `xi_nl` tables that half the downstream modules rely
on, plus an *optional* set of one-halo and two-halo lensing profiles
(`Sigma_nfw`, `dSigma_nfw`, `Sigma_hh`, `dSigma_hh`, `Wp_hh`). Its core
products:

- **Halo bias $b(M, z)$** — on a log-spaced mass grid, the peak height
  $\nu(M) = \delta_c/\sigma(M, z{=}0)$ is computed via
  `cluster_toolkit.peak_height.nu_at_M`, then for each redshift the
  Tinker 2010 bias formula is evaluated at $\nu(M)/(D(z)/D(0))$. The
  explicit $D(0)$ division matters: the CosmoSIS growth module publishes
  an un-normalised $D(z)$ with $D(0) \neq 1$, and prior to May 2026 the
  un-normalised value inflated the effective $\nu$ by
  $1/D(0) \simeq 1.32$ (at $z = 0.425$, $D(0) \simeq 0.758$), producing
  halo biases up to $2\times$ too large.
- **Nonlinear correlation $\xi_{\rm NL}(r, z)$** — via
  `ct.xi.xi_mm_at_r` over $r \in [10^{-3}, 10^{3}]\,h^{-1}\,\mathrm{Mpc}$
  (128 nodes) at each redshift of the linear power grid.

The two lensing branches are gated by separate ini flags:

- **`compute_lensing_1h`** — analytic NFW $\Sigma(R, M)$ and
  $\Delta\Sigma(R, M)$ with the Child-18 concentration, written to
  `haloModel/{Sigma_nfw, dSigma_nfw, concentration}`. This is the centred
  profile that feeds the 1-halo branch: `Shear1hMisSel` needs it for the
  centred component of its miscentering mixture.
- **`compute_lensing_2h`** — the two-halo term via
  `ct_2hTerm.pk_to_dsigma`, a redshift loop of cluster_toolkit Hankel
  transforms ($P \to \xi \to \Sigma \to \Delta\Sigma$), written to
  `haloModel/{Sigma_hh, dSigma_hh, Wp_hh}`. Together with $b(M, z)$ these
  are the ingredients of the fiducial two-halo lensing composition,
  $\mathrm{Shear} = 1h + 2h$: the halo-halo profiles
  $\Sigma_{\rm hh}$/$\Delta\Sigma_{\rm hh}$/$W_{p,\rm hh}$ carry the
  matter correlation, scaled by the halo bias to give the two-halo
  contribution around a halo of mass $M$.

The original module always ran both branches; the split was introduced
because the 2h Hankel loop costs $\sim 200$–$300$ ms/sample at
$N_z = 50$ and dominates the module's runtime. In the Costanzi-2026
production pipeline the modern two-halo projection branch (`shear_prj`)
does *not* read the 2h outputs — only the legacy `ShearTotSel` family
(the fiducial 1h+2h composition) does — so production sets

```
compute_lensing_1h = T
compute_lensing_2h = F    # saves 180-200 ms
```

a pure skip with no accuracy cost for the projection pipeline. Pipelines
that build the fiducial 1h+2h shear must instead keep
`compute_lensing_2h = T` so that `Sigma_hh`, `dSigma_hh`, and `Wp_hh` are
available downstream.

*Source: [`docs/pipeline_modules.tex`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/docs/pipeline_modules.tex) §`halo_model`: bias $b(M,z)$ and
$\xi_{\rm NL}(r,z)$.*

### Comparing the two compositions

```{admonition} Figure removed — orange curve used the wrong composition
:class: warning
A figure previously here compared the reference (blue) against a
labeled "fiducial 1h+2h" curve computed as a plain **sum**,
$\langle\Delta\Sigma^{1h}\rangle_i + \langle b\rangle_i\,\Delta\Sigma_{\rm 2h}$.
That contradicts this project's own documented standard
({doc}`../observables/second_halo_term`): the traditional composition is
the **pointwise max**, $\Sigma_{\max} = \max(\Sigma_{\rm NFW},\,
b\,\Sigma_{\rm hh})$ (Hayashi & White 2008, the DES Y1 prescription),
implemented by
[`shear1h2h_max.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/pipelines/des_y3/observables/shear_1h2h/fast_mass/python/shear1h2h_max.py).
Removed rather than left showing the wrong physics. Regenerating it
needs a real fiducial pipeline dump with `compute_lensing_2h = T` — not
available in this environment; see {doc}`../observables/second_halo_term`
for the correct recipe and what a regeneration needs.
```

Two implementation findings from producing this figure (2026-08-10):

- The **wired legacy 1h+2h modules are currently broken**: with
  `compute_lensing_2h = T`, `haloModel/dSigma_hh` comes out NaN for
  $R \lesssim 8.6\,h^{-1}$cMpc, and `DSigmaTotSel`/`SigmaTotSel`
  interpolate the 2h table on the wrong radial axis (`r_sigma`,
  0.1–20, instead of the `Rp` grid, 1–35, the table is computed on).
  The fiducial curve above is therefore assembled from
  $\langle b\rangle_i$ and $\xi_{\rm NL}$ directly; fixing the
  `*TotSel` path is future code work (see the
  {doc}`status appendix <../modules/historical>`).
- The historical combination
  $\mathrm{ShearCorr}(R) = B_{\rm prj}(R)\,[\mathrm{Shear}_{1h} +
  \mathrm{Shear}_{2h}]$ is **not implemented anywhere in the current
  code** (no `ShearCorr`/`B_prj` symbol exists in `src/` or
  `y3_buzzard/`); it is recorded here as terminology from earlier
  planning only.

## Selection functions

This chapter defines the richness–mass relation, the observed-richness
projection kernel, and the bin-integrated selection kernels that together
build the selection function $S_{ij}(\ln M, z)$ used by every population
integral in the pipeline.

### Intrinsic richness–mass relation

This section defines the two models used for the intrinsic (mass–richness)
relation $P(\lambda^{\mathrm{tr}} \mid M, z)$ — the probability that a halo
of mass $M$ at true redshift $z^{\mathrm{tr}}$ hosts a true richness
$\lambda^{\mathrm{tr}}$.

#### Log-normal model

Following Costanzi et al. (2021, their Eq. 2), the mean of $\ln\lambda$ at
fixed mass and redshift is a linear combination of $\ln M$ and
$\ln[(1+z)/(1+z_p)]$,

$$
\langle \ln\lambda \rangle(M,z)
=
\ln A_\lambda
+ B_\lambda \ln\!\left(\frac{M}{M_p}\right)
+ C_\lambda \ln\!\left(\frac{1+z}{1+z_p}\right),
$$

with pivot mass $M_p = 3\times 10^{14}\,h^{-1}M_\odot$ and pivot redshift
$z_p = 0.45$. The intrinsic scatter is log-normal, with log-space variance

$$
\sigma^2_{\ln\lambda}(M,z)
= D_\lambda^{2}
+ \frac{\langle\lambda\rangle - 1}{\langle\lambda\rangle^{2}},
$$

where the second term is the Poisson-like contribution from the discrete
galaxy counts. The mass–richness PDF is then

$$
P(\lambda^{\mathrm{tr}} \mid M, z)
=
\frac{1}{\lambda^{\mathrm{tr}}\,\sigma_{\ln\lambda}\sqrt{2\pi}}
\exp\!\left[
-\frac{\left(\ln\lambda^{\mathrm{tr}}-\langle\ln\lambda\rangle\right)^{2}}{2\,\sigma^2_{\ln\lambda}}
\right],
$$

i.e. $\ln\lambda^{\mathrm{tr}} \sim \mathcal N\!\left(\langle\ln\lambda\rangle,\,
\sigma^{2}_{\ln\lambda}\right)$. The linear-space moments are the standard
log-normal ones,

$$
\langle\lambda\rangle
= \exp\!\left(\langle\ln\lambda\rangle+\tfrac{1}{2}\sigma^{2}_{\ln\lambda}\right),
\qquad
\sigma_{\lambda}^{2}
= \langle\lambda\rangle^{2}\left(e^{\sigma^{2}_{\ln\lambda}}-1\right),
$$

so the fractional scatter is $\sigma_\lambda/\langle\lambda\rangle =
\sqrt{e^{\sigma^2_{\ln\lambda}}-1} \approx \sigma_{\ln\lambda}$ for small
scatter.

| Parameter | Meaning |
|---|---|
| $A_\lambda$ | amplitude of the mean mass–richness relation |
| $B_\lambda$ | mass slope |
| $C_\lambda$ | redshift evolution |
| $D_\lambda$ | intrinsic halo-to-halo log-scatter |
| $M_p$ | pivot mass, $3\times 10^{14}\,h^{-1}M_\odot$ |
| $z_p$ | pivot redshift, $0.45$ |

#### Shifted-Poisson HOD model (production choice)

The DES Y1 analysis (Costanzi et al. 2019b, with weak-lensing mass
calibration from McClintock et al. 2019) adopts a halo-occupation
distribution (HOD) model in which the intrinsic richness splits into a
central and a satellite component,
$\lambda^{\mathrm{tr}} = \lambda^{\mathrm{cen}} + \lambda^{\mathrm{sat}}$,
with $\lambda^{\mathrm{cen}} = 1$ above the detection threshold
$M_{\mathrm{min}}$ and zero below. The mean satellite richness follows the
power-law form later used in Costanzi et al. (2026),

$$
\langle\lambda^{\mathrm{sat}}\rangle(M,z)
=
\left(\frac{M-M_{\mathrm{min}}}{M_1-M_{\mathrm{min}}}\right)^{\!\alpha}
\left(\frac{1+z}{1+z_\star}\right)^{\!\epsilon},
$$

where $M_1$ is the halo mass at which a halo hosts on average one satellite
and $z_\star = 0.45$ is a pivot redshift. Because satellite counts are
intrinsically discrete, the stochasticity is Poissonian at low occupancy
$\langle\lambda^{\mathrm{sat}}\rangle \lesssim 10$ (where the log-normal
overestimates the scatter) and super-Poissonian at high occupancy due to an
additional halo-to-halo term $\sigma_{\mathrm{intr}}$, giving the
two-component variance

$$
\sigma^{2}_{\lambda^{\mathrm{tr}} \mid M}
\simeq
\langle\lambda^{\mathrm{sat}}\rangle
+
\left(\sigma_{\mathrm{intr}}\,\langle\lambda^{\mathrm{sat}}\rangle\right)^{2}.
$$

The exact $P(\lambda^{\mathrm{tr}} \mid M, z)$ — a Poisson distribution
convolved with a Gaussian halo-to-halo scatter — has no closed form;
Costanzi et al. (2019b) approximated it by a skew-normal with parameters
from a pre-computed lookup table. The cleaner alternative used here
(priv. comm. with M. Costanzi) approximates the Poisson-plus-scatter law by
a continuous *shifted-Poisson* distribution. Define

$$
\delta = \big(\sigma_{\mathrm{intr}}\,\langle\lambda^{\mathrm{sat}}\rangle\big)^{2},
\qquad
\nu = \langle\lambda^{\mathrm{sat}}\rangle + \delta,
$$

where $\delta$ is the intrinsic halo-to-halo variance contribution and
$\nu$ is the shifted Poisson rate. Then

$$
P(\lambda^{\mathrm{tr}} \mid M, z)
=
\exp\!\Big[
-\nu
+(\lambda^{\mathrm{tr}}+\delta-1)\ln\nu
-\ln\Gamma(\lambda^{\mathrm{tr}}+\delta)
\Big],
$$

obtained by promoting the factorial of the Poisson PMF to a gamma function
and shifting its argument by $\delta$. This keeps the two-component
variance scaling above and recovers the pure Poisson limit as
$\sigma_{\mathrm{intr}} \to 0$. Numerical comparisons show it tracks the
exact Poisson-convolved-with-Gaussian law essentially everywhere, including
the low-$\lambda^{\mathrm{tr}}$ tail where the skew-normal breaks down. For
forward-model use it is closed-form (no lookup tables, fully differentiable
in $(M,z)$) and extends smoothly to the non-integer richness values
required by the quadrature.

To leading order its effective moments are

$$
\mu_{\mathrm{eff}} \approx \langle\lambda^{\mathrm{sat}}\rangle,
\qquad
\sigma_{\mathrm{eff}}^{2}
\approx
\langle\lambda^{\mathrm{sat}}\rangle
+
\left(\sigma_{\mathrm{intr}}\,\langle\lambda^{\mathrm{sat}}\rangle\right)^{2}.
$$

| Parameter | Meaning |
|---|---|
| $M_{\mathrm{min}}$ | detection threshold: minimum mass to host a central |
| $M_1$ | mass at which a halo hosts on average one satellite |
| $\alpha$ | mass slope of the satellite occupation |
| $\epsilon$ | redshift evolution |
| $z_\star$ | pivot redshift, $0.45$ |
| $\sigma_{\mathrm{intr}}$ | super-Poissonian halo-to-halo scatter |

*Source: [`RichnessSelection/docs/richness_selection_function.tex`](https://github.com/estevesjh/RichnessSelection/blob/main/docs/richness_selection_function.tex) §Models
for the mass–richness relation.*

### Observed richness: the projection kernel

This section defines the observational kernel
$P(\lambda^{\mathrm{ob}} \mid \lambda^{\mathrm{tr}}, z)$ mapping intrinsic
to measured richness, including measurement noise and projection effects
(Costanzi et al. 2019a, Eqs. 3, 5, 6; re-used unchanged in Costanzi et al.
2019b, 2021).

The derivation starts from the additive decomposition

$$
\lambda^{\mathrm{ob}} = \lambda^{\mathrm{tr}} + \Delta^{\mathrm{bkg}} + \Delta^{\mathrm{prj}},
$$

where $\Delta^{\mathrm{bkg}}$ is a Gaussian background/measurement
fluctuation,

$$
\Delta^{\mathrm{bkg}} \sim \mathcal N(\Delta\mu, \sigma^2),
$$

and $\Delta^{\mathrm{prj}} \ge 0$ is a one-sided projection boost modelled
as a spike at zero plus an exponential tail,

$$
P\!\left(\Delta^{\mathrm{prj}} \mid \lambda^{\mathrm{tr}}, z\right)
=
(1-f^{\mathrm{prj}})\,\delta_{\mathrm{D}}(\Delta^{\mathrm{prj}})
+
f^{\mathrm{prj}}\,\tau\,e^{-\tau\,\Delta^{\mathrm{prj}}}\,\Theta(\Delta^{\mathrm{prj}}),
$$

with $\delta_{\mathrm{D}}$ the Dirac delta and $\Theta$ the Heaviside step
function. Convolving the Gaussian noise with the projection boost produces
a classic *exponentially modified Gaussian* (EMG, or "ex-Gaussian") — the
law of a sum $X = G + E$ with $G \sim \mathcal N$ and $E \sim \mathrm{Exp}$
— the standard model for a Gaussian core plus a one-sided exponential tail
(Grushka 1972). The resulting kernel is

$$
\begin{aligned}
P(\lambda^{\mathrm{ob}} \mid \lambda^{\mathrm{tr}}, z)
&=
(1-f^{\mathrm{prj}})\,\mathcal N\!\left(\lambda^{\mathrm{ob}};\, \mu, \sigma\right) \\
&\quad
+
f^{\mathrm{prj}}\,
\frac{\tau}{2}
\exp\!\left[
\frac{\tau}{2}\left(2\mu+\tau\sigma^{2}-2\lambda^{\mathrm{ob}}\right)
\right]
\operatorname{erfc}\!\left(
\frac{\mu+\tau\sigma^{2}-\lambda^{\mathrm{ob}}}{\sqrt{2}\,\sigma}
\right),
\end{aligned}
$$

with

$$
\mu \equiv \lambda^{\mathrm{tr}} + \Delta\mu.
$$

The first line is the "background" (BKG-only) limit recovered when
$f^{\mathrm{prj}} = 0$ — a pure Gaussian
$P_{\mathrm{bkg}}(\lambda^{\mathrm{ob}} \mid \lambda^{\mathrm{tr}})$ as used
in Costanzi et al. (2021). The second line is the EMG projection (PRJ)
contribution; in Costanzi et al. (2021) this term also absorbs residual
masking/percolation effects. The kernel is *not* normalised to unity over
$\lambda^{\mathrm{ob}} \in \mathbb R$ for $f^{\mathrm{prj}} > 0$ only
because $\Delta^{\mathrm{prj}} \ge 0$; total probability is conserved by
construction through the decomposition above.

The four kernel coefficients $\{\Delta\mu, \sigma, f^{\mathrm{prj}}, \tau\}$
all depend on $(\lambda^{\mathrm{tr}}, z)$ and are calibrated empirically
(Costanzi et al. 2019a use synthetic-cluster injections in SDSS):

| Coefficient | Meaning |
|---|---|
| $\Delta\mu(\lambda^{\mathrm{tr}}, z)$ | mean bias of the background/measurement Gaussian; typically $\Delta\mu < 0$ because redMaPPer's global background subtraction biases $\lambda^{\mathrm{ob}}$ low |
| $\sigma(\lambda^{\mathrm{tr}}, z)$ | Gaussian width combining photometric noise and uncorrelated background fluctuations |
| $f^{\mathrm{prj}}(\lambda^{\mathrm{tr}}, z) \in [0,1]$ | fraction of clusters affected by a projection boost (line-of-sight overlap with other haloes); increases with $\lambda^{\mathrm{tr}}$ and $z$ |
| $\tau(\lambda^{\mathrm{tr}}, z) > 0$ | inverse scale of the exponential projection tail: smaller $\tau$ means longer tails and stronger projections |

*Source: [`RichnessSelection/docs/richness_selection_function.tex`](https://github.com/estevesjh/RichnessSelection/blob/main/docs/richness_selection_function.tex)
§Closed-form of the observed richness kernel with projection effects.*

### The observed-richness kernel

This section defines the closed-form observed-richness kernel
$\mathcal S_i(\lambda^{\mathrm{tr}}, z)$ — the probability, at fixed
$\lambda^{\mathrm{tr}}$, of being observed inside the richness bin
$\Delta\lambda_i \equiv [\lambda_i^{\min}, \lambda_i^{\max}]$ — and the
Gauss–Legendre quadrature that assembles the richness selection function
$S_i(M, z^{\mathrm{tr}})$.

#### Definition and Gaussian/EMG split

The bin-integrated observational kernel is

$$
\mathcal S_i(\lambda^{\mathrm{tr}}, z)
\equiv
\int_{\lambda_i^{\min}}^{\lambda_i^{\max}}
d\lambda^{\mathrm{ob}}\,
P(\lambda^{\mathrm{ob}} \mid \lambda^{\mathrm{tr}}, z).
$$

Inserting the projection kernel and exchanging the sum with the bin
integral yields

$$
\mathcal S_i(\lambda^{\mathrm{tr}}, z)
=
(1-f^{\mathrm{prj}})\,\mathcal S_i^{\mathrm{G}}
+
f^{\mathrm{prj}}\,\mathcal S_i^{\mathrm{EMG}},
$$

where $\mathcal S_i^{\mathrm{G}}$ and $\mathcal S_i^{\mathrm{EMG}}$ are the
bin integrals of the Gaussian and EMG components respectively. Each piece
integrates analytically.

#### Gaussian piece: CDF differencing

The Gaussian term is the integral of a normal PDF between two limits,

$$
\mathcal S_i^{\mathrm{G}}
=
\Phi\!\left(\frac{\lambda_i^{\max}-\mu}{\sigma}\right)
-
\Phi\!\left(\frac{\lambda_i^{\min}-\mu}{\sigma}\right),
$$

where $\Phi$ is the standard normal CDF and the arguments are the
standardised bin edges relative to the mean
$\mu = \lambda^{\mathrm{tr}} + \Delta\mu$.

#### EMG piece: the CDF of $X = G + E$

The EMG contribution is the difference of the EMG CDF at the two bin edges,

$$
\mathcal S_i^{\mathrm{EMG}}
=
F_{\mathrm{EMG}}(\lambda_i^{\max}; \mu, \sigma, \tau)
-
F_{\mathrm{EMG}}(\lambda_i^{\min}; \mu, \sigma, \tau),
$$

with the closed-form CDF

$$
F_{\mathrm{EMG}}(x; \mu, \sigma, \tau)
=
\Phi\!\left(\frac{x-\mu}{\sigma}\right)
-
\exp\!\left[-\tau(x-\mu)+\tfrac{1}{2}\tau^{2}\sigma^{2}\right]
\Phi\!\left(\frac{x-\mu}{\sigma}-\tau\sigma\right).
$$

The first term is the Gaussian CDF at $x$; the second encodes the
correction from the exponential projection tail. (Derivation — write
$X = G + E$ with $G \sim \mathcal N(\mu, \sigma^2)$,
$E \sim \mathrm{Exp}(\tau)$, condition on $G$, and complete the square
— in the appendix of the optical-selection paper and
[`RichnessSelection/docs/richness_selection_function.tex`](https://github.com/estevesjh/RichnessSelection/blob/main/docs/richness_selection_function.tex).)

#### Assembled closed form

With the definite-integral notation
$\left.g(\lambda^{\mathrm{ob}})\right|_{\Delta\lambda_i} \equiv
g(\lambda_i^{\max}) - g(\lambda_i^{\min})$, the compact form is

$$
\mathcal S_i(\lambda^{\mathrm{tr}}, z)
=
(1-f^{\mathrm{prj}})\left.\Phi\!\left(\frac{\lambda^{\mathrm{ob}}-\mu}{\sigma}\right)\right|_{\Delta\lambda_i}
+
f^{\mathrm{prj}}\,\Big.F_{\mathrm{EMG}}(\lambda^{\mathrm{ob}}; \mu, \sigma, \tau)\Big|_{\Delta\lambda_i},
$$

and substituting $F_{\mathrm{EMG}}$ and
$\mu = \lambda^{\mathrm{tr}} + \Delta\mu$ explicitly gives the fully
expanded closed form

$$
\mathcal S_i(\lambda^{\mathrm{tr}}, z)
=
\left.\Phi\!\left(\frac{\lambda^{\mathrm{ob}}-\lambda^{\mathrm{tr}}-\Delta\mu}{\sigma}\right)\right|_{\Delta\lambda_i}
-\,
f^{\mathrm{prj}}\left.
\exp\!\left(-\tau(\lambda^{\mathrm{ob}}-\lambda^{\mathrm{tr}}-\Delta\mu)+\tfrac{1}{2}\tau^{2}\sigma^{2}\right)
\Phi\!\left(\frac{\lambda^{\mathrm{ob}}-\lambda^{\mathrm{tr}}-\Delta\mu}{\sigma}-\tau\sigma\right)
\right|_{\Delta\lambda_i}.
$$

The latent-richness dependence is explicit: $\lambda^{\mathrm{tr}}$ enters
every Gaussian argument through the shifted residual
$\lambda_i^{\min/\max} - \lambda^{\mathrm{tr}} - \Delta\mu$.

#### Gauss–Legendre quadrature and the $S_i(\ln M, z)$ assembly

With $\mathcal S_i$ known analytically, the richness selection function
reduces to a single integral against the mass–richness PDF,

$$
S_i(M, z^{\mathrm{tr}})
=
\int_0^\infty d\lambda^{\mathrm{tr}}\,
\mathcal S_i(\lambda^{\mathrm{tr}}, z^{\mathrm{tr}})\,
P(\lambda^{\mathrm{tr}} \mid M, z^{\mathrm{tr}}),
$$

approximated by Gauss–Legendre quadrature on a finite interval
$[a,b] \subset (0,\infty)$:

$$
S_i(M, z^{\mathrm{tr}})
\approx
\sum_{k=1}^{N_q}
W_k\,
\mathcal S_i(\lambda_k, z^{\mathrm{tr}})\,
P(\lambda_k \mid M, z^{\mathrm{tr}}),
\qquad
\lambda_k=\frac{b-a}{2}\,t_k+\frac{a+b}{2},
\quad
W_k=\frac{b-a}{2}\,w_k,
$$

where $(t_k, w_k)$ are the standard Gauss–Legendre nodes and weights on
$[-1,1]$, fixed once and for all; only the interval $[a,b]$ depends on the
model parameters. Since $\lambda^{\mathrm{tr}} > 0$, the limits bracket the
support of $P(\lambda^{\mathrm{tr}} \mid M, z)$,

$$
a=\max\!\big(0,\, \mu_{\mathrm{eff}}-L\,\sigma_{\mathrm{eff}}\big),
\qquad
b=\mu_{\mathrm{eff}}+L\,\sigma_{\mathrm{eff}},
$$

with $(\mu_{\mathrm{eff}}, \sigma_{\mathrm{eff}})$ the mean and standard
deviation of the mass–richness relation and $L \sim 6$–$8$ chosen so the
interval captures essentially all of the probability mass. For the
log-normal model $(\mu_{\mathrm{eff}}, \sigma_{\mathrm{eff}}) =
(\langle\lambda\rangle, \sigma_\lambda)$ from the log-normal moments; for
the shifted-Poisson HOD model they are the effective moments
$\mu_{\mathrm{eff}} \approx \langle\lambda^{\mathrm{sat}}\rangle$,
$\sigma_{\mathrm{eff}}^2 \approx \langle\lambda^{\mathrm{sat}}\rangle +
(\sigma_{\mathrm{intr}}\langle\lambda^{\mathrm{sat}}\rangle)^2$. Different
mass–richness models change $(\mu_{\mathrm{eff}}, \sigma_{\mathrm{eff}})$
but not the structure of the sum.

Plugging $S_i$ back into the forward-model number counts collapses the
original 5D integral over
$(M, z^{\mathrm{tr}}, \lambda^{\mathrm{tr}}, \lambda^{\mathrm{ob}}, z^{\mathrm{ob}})$
to a 2D integral over $(M, z^{\mathrm{tr}})$:

$$
\langle N_{ij}\rangle
=
\int dM \int dz^{\mathrm{tr}}\;
\Omega(z^{\mathrm{tr}})\,\frac{dV}{d\Omega\,dz^{\mathrm{tr}}}\,n(M, z^{\mathrm{tr}})\;
S_i(M, z^{\mathrm{tr}})\,\mathcal S_j(z^{\mathrm{tr}}),
$$

where the observed-redshift kernel factorises as a difference of normal
CDFs,
$\mathcal S_j(z^{\mathrm{tr}}) =
\left.\Phi\!\left((z^{\mathrm{ob}}-z^{\mathrm{tr}})/\sigma_z\right)\right|_{\Delta z_j}$,
with bin-dependent photo-$z$ scatter $\sigma_z \equiv \sigma_z(\Delta\lambda_i)$.
The evaluation grid in $(\ln M, z)$ then needs only elementary special
functions ($\Phi$, $\exp$, $\ln\Gamma$), with the Gauss–Legendre nodes and
weights pre-computed.

*Source: [`RichnessSelection/docs/richness_selection_function.tex`](https://github.com/estevesjh/RichnessSelection/blob/main/docs/richness_selection_function.tex)
§Closed-form of the observed richness kernel with projection effects,
§Gauss–Legendre numerical integration, §Summary.*
