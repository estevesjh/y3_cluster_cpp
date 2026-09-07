# Boost Factor

`Python` · `y3_cluster_cpp` (`src/pipelines/systematics/boost_factor/`) · `Systematics` · module `boost_factor` · `<1 ms/sample` · **published, not yet consumed**

Publishes the **boost factor** $B(R)$ — the correction for the dilution
of the lensing signal by cluster member galaxies mistaken for
background sources — for each of the 12 $(\lambda, z)$ bins, from
externally calibrated per-bin parameters. The model and its calibration
follow the DES Y1 stacked-lensing analysis
([McClintock et al. 2019, MNRAS 482, 1352](https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1352M/abstract),
arXiv:[1805.00039](https://arxiv.org/abs/1805.00039)); the module was
written by Arwa Abdulghafour for the GPU projection pipeline.

## Script

- CosmoSIS module: [`src/pipelines/systematics/boost_factor/apply_boost_factor.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/systematics/boost_factor/apply_boost_factor.py)
  — evaluates $B(R)$ on the configured radii from the fixed
  `(rs, b0)` pair of every bin and writes the `boost_factor` section.
- Model + calibration code: [`bf_likelihood_improved.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/systematics/boost_factor/bf_likelihood_improved.py)
  (`boost_factor_model`, the Y1 data loaders, and a standalone
  $\chi^2$ likelihood).
- Offline calibration: `fit_y1_bins.py` (all 12 bins in one go) and the
  standalone `bf_pipeline_improved.ini` + `bf_values_all_bins.ini`
  emcee pipeline, run against the DES Y1 boost-factor profiles
  (`full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost{,_cov}.dat`).
  These produce the `rs_l*_z*`/`b0_l*_z*` numbers; they are not part of
  the sampling pipeline.

## Physics

A fraction of the galaxies in the source catalogue behind a cluster
are in fact cluster members with no lensing signal. Their photo-$z$
scatter puts them in the source sample, they are concentrated towards
the cluster centre, and they dilute the measured mean tangential shear
by the local contamination fraction $f_{\rm cl}(R)$:

$$\Delta\Sigma_{\rm obs}(R) = \frac{\Delta\Sigma_{\rm true}(R)}{B(R)},
\qquad B(R) = \frac{1}{1 - f_{\rm cl}(R)} \ge 1 .$$

$B(R)$ is measured from the source $p(z)$ decomposition around clusters
versus random points and modelled as the projected NFW shape of the
member-galaxy distribution,

$$B(R) = 1 + b_0\,\frac{1 - f(x)}{x^2 - 1}, \qquad x = \frac{R}{r_s},
\qquad
f(x) = \begin{cases}
\dfrac{\arctan\sqrt{x^2-1}}{\sqrt{x^2-1}}, & x > 1,\\[8pt]
\dfrac{\operatorname{arctanh}\sqrt{1-x^2}}{\sqrt{1-x^2}}, & x < 1,\\[8pt]
1, & x = 1 \;\;(B \to 1 + b_0/3),
\end{cases}$$

with two parameters per bin: the amplitude $b_0$ and the scale radius
$r_s$ (same units as $R$). The implementation evaluates the removable
singularity at $x = 1$ analytically.

The boost is a **source-catalogue** correction: it is independent of
the one-halo/two-halo (or one-halo/projection) decomposition of the
theory and must be applied once, to the combined theory shear, by
*dividing* — the model is diluted the same way the raw data are, or
equivalently the data are boosted. In DES Y1 the fit was performed
jointly with the lensing profile; here $(r_s, b_0)$ are frozen at their
offline best fit and treated as fixed inputs.

## CosmoSIS setup

```ini
[boost_factor]
file  = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/systematics/boost_factor/apply_boost_factor.py
radii = <shear radii, same units as rs>
```

and, in the values file, one fixed pair per bin:

```ini
[boost_factor]
rs_l0_z0 = 1.0
b0_l0_z0 = 0.3
; ... rs_l3_z2, b0_l3_z2
```

- Ordering: anywhere after `consistency`; it reads only the values
  section. Its output is not read by any module of the reference
  pipeline yet.
- Bin indices: `l0…l3` are the four richness bins, `z0…z2` the three
  redshift bins, matching the `sel_function` wall.

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `radii` | radii on which $B(R)$ is evaluated (the shear wall radii) | as $r_s$ | 15-radius wall |

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `boost_factor/rs_l{l}_z{z}` | scale radius per bin | as `radii` | values file (fixed) |
| `boost_factor/b0_l{l}_z{z}` | amplitude per bin | — | values file (fixed) |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `boost_factor/R` | the radii | as input | — |
| `boost_factor/B_l{l}_z{z}` | $B(R)$ per bin | dimensionless, `(N_R,)` | **nothing yet** |

```{admonition} Status
:class: warning
The module publishes $B(R)$ but no consumer divides the theory shear by
it: `likelihood_cp.py` ({doc}`../observables/likelihood`) does not read
`boost_factor/*`, and the per-bin `(rs, b0)` values in the driving
values files are placeholders until `fit_y1_bins.py` has been run
against the Y1 data. Wiring the division into the likelihood's shear
theory is the pending step.
```
