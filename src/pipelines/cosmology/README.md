# Cosmology and halo-model physics

This directory contains the survey-independent cosmological and halo-model
ingredients used by the cluster-observable pipelines. These models describe
the halo population, the lensing profile of an individual halo, the effect of
halo bias on correlated structure along the line of sight, the modification of
that bias by optical selection, and the lensing geometry that converts surface
density into shear.

The consumers are the observable implementations under
`src/pipelines/des_y3/` and future survey pipelines. This directory contains
the physics layer, not the final number-count or lensing integrals.

## Physical role

The main dependency chain is:

```text
halo mass M, redshift z
        │
        ├── halo bias b(M,z)
        ├── concentration c(M,z)
        │       └── NFW Sigma(R), DeltaSigma(R)
        ├── selection bias b_sel(theta)
        │       └── projection lensing from correlated line-of-sight halos
        └── Sigma_crit^-1(z_lens, source)
                └── converts DeltaSigma into tangential shear
```

For the cluster observables, these ingredients are combined with the halo
mass function, richness and redshift selection kernels, survey geometry, and
mass/redshift integration strategies implemented elsewhere in the pipeline.

## Models implemented here

| Physical ingredient | File | Model and role |
| --- | --- | --- |
| Halo bias and halo-model lensing | `halo_model.py` | Tinker et al. (2010) halo bias; one-halo NFW and two-halo lensing terms; cosmology rescaling through `scaleShiftCosmo` |
| Mass--concentration relations | `concentration.py` | Child18 and Duffy relations; peak-height quantities; concentration evaluated under the relevant halo-mass definition |
| Mass-definition conversion | `hydro_mc.py` | Vendored Ragagnin et al. (2020) conversion between M200m/M200c and the corresponding concentrations |
| Analytic NFW profile | `nfw_model.py` | Wright & Brainerd (2000) surface density Sigma and excess surface density DeltaSigma |
| Projection-lensing coefficients | `prj_params.py` | Frozen Costanzi et al. (2026) EMG projection-kernel coefficients |
| Selection-affected halo bias | `bsel.py` | Exact-wall selection-bias closure producing the small- and large-angle bias plateaus used by the projection branch |
| Lensing geometry | `sigma_crit_inv.py` | Sigma_crit^-1 as a function of lens redshift and source distribution, using the beta lookup table and cosmological shift |

## Observable connections

### One-halo lensing

The one-halo term describes the matter profile of the target cluster. The
centered and miscentered profiles are built from the NFW surface-density
model, the halo mass, the concentration relation, and the target-cluster
miscentering model. The observable pipeline then averages this profile over
the selected cluster population and multiplies it by Sigma_crit^-1.

### Two-halo and projection lensing

The two-halo contribution is sourced by matter in correlated neighboring
halos and the surrounding large-scale structure. It depends on halo bias,
the nonlinear matter correlation function, the NFW profile of neighboring
halos, and the projection geometry.

The projection branch additionally uses the Costanzi et al. (2026) EMG
projection kernel and the selection-affected bias `b_sel`. This is distinct
from the conventional two-halo term evaluated with the unselected halo bias.

### Optical selection bias

`bsel.py` computes the bias of the halo population selected by an observed
richness and observed-redshift wall. It consumes the shared richness
selection inputs (`lambda_edges` and `PHOD`) and produces the bias closure
used by the projection-lensing calculation.

## Physical conventions

- Halo masses may use either M200c or M200m. The mass definition must match
  the concentration relation and the mass-conversion path being used.
- The profile normalization is based on the present-day mean matter density,
  `rho_m0 = Omega_m0 * rho_crit,0`, in comoving coordinates. It has no extra
  redshift-density factor.
- The one-halo profile uses the concentration evaluated at the cluster
  redshift. The projection profile uses the configured mass--concentration
  relation rather than an arbitrary fixed concentration.
- `b_sel` is the selection-affected bias used for correlated line-of-sight
  structure. It is not the same quantity as the ordinary halo bias
  `b(M,z)`.
- Sigma, DeltaSigma, and Sigma_crit^-1 must be combined with consistent
  radius, mass, distance, and comoving/physical units. The consumer modules
  define the final observable normalization.

## What is not implemented here

This directory is not a complete cosmology package. In particular:

- The halo mass function is supplied by the shared/CosmoSIS layers; it is not
  implemented here.
- Richness and photometric-redshift selection kernels live under
  `src/pipelines/shared/` and the survey-specific pipeline layers.
- Number-count, one-halo, and projection-lensing integrations live under
  `src/pipelines/des_y3/` or another survey's observable implementation.
- Survey-specific calibration data and CosmoSIS run-management
  configuration live outside this directory.

## Scientific provenance

The implementation is based on the following models and references:

- DES Cluster et al. (2023), the cluster number-count and population-averaged
  lensing forward model and CosmoSIS software framework
  ([arXiv:2309.06593](https://arxiv.org/abs/2309.06593))
- Tinker et al. (2010), halo bias
- Wright & Brainerd (2000), analytic NFW lensing profiles
- Costanzi et al. (2026), optical selection bias and projection lensing
  ([arXiv:2604.05833](https://arxiv.org/abs/2604.05833))
- Child18 and Duffy mass--concentration relations
- Ragagnin et al. (2020), the vendored mass-definition conversion library in
  `hydro_mc.py`

## Implementation notes

### Import convention

Add `<repo>/src/pipelines` to `sys.path`, then import `cosmology` as a
top-level package:

```python
from cosmology.halo_model import lensingModel
from cosmology.concentration import child18_mass_concentration
from cosmology.prj_params import PrjParams
```

Do not import through `pipelines.cosmology`: `src/pipelines` is a namespace
root without an `__init__.py` and is not imported as the `pipelines` package.

### Relationship to `y3_buzzard/`

The modules here are the canonical source for new pipeline code. The original
modules under `y3_buzzard/` remain untouched because CosmoSIS configurations
in sibling repositories may still import them by path. Existing pipelines
therefore continue to use the old paths, while new survey implementations
import from `cosmology`.
