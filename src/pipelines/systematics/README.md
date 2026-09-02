# Systematics implementations

This directory owns the selection-systematics pieces used by the DES Y3
observable pipelines. It is the canonical maintained location for these
models; the older copies under `src/pipelines/shared/`, `src/pipelines/cosmology/`,
and `src/models/` remain available as compatibility references.

```text
systematics/
├── selection_richness/python/
│   ├── sel_function.py       # S_ij(ln M, z)
│   └── sel_kernels.py        # offline/reference selection helpers
├── selection_bias/
│   ├── python/bsel.py        # b_small, b_large from P1, I1, J
│   └── cpp/bsel_bins_t.hh    # exact wall-row lookup for C++ consumers
├── selection_function/python/
│   └── prj_params.py         # Costanzi projection-kernel coefficients
├── costanzi_bprj/
│   ├── python/costanzi_bprj.py   # B_prj(R) selection-bias correction (pydantic params)
│   └── cpp/costanzi_bprj_t.hh    # same model, DataBlock constructor
└── shear_prj/cpp/
    ├── sigma_prj_t.hh                # exact-z projection shear core
    ├── sigma_prj_frozen_t.hh          # frozen-physics projection backend
    └── sigma_prj_frozen_interp_t.hh  # continuous frozen Cuhre backend
```

The DES Y3 pipeline imports the Python selection modules from here and its
projection-shear C++ driver includes the `shear_prj` core from here. The
lower-level numerical and datablock utilities remain in the sibling
`../shared/` package; halo-model physics remains in `../cosmology/`.

The live configurations are:

```ini
[sel_function]
file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/systematics/selection_richness/python/sel_function.py

[bsel]
file = ${Y3_CLUSTER_CPP_DIR}/src/pipelines/systematics/selection_bias/python/bsel.py
```

`prj_params.py` is loaded explicitly by configurations that need to publish
the projection coefficients. No old copy is deleted or silently redirected.

## Costanzi-2026 `B_prj(R)` correction (`costanzi_bprj/`)

Appendix C of Costanzi et al. (2026), arXiv:2604.05833: the optical selection
bias on the projected density profile is a multiplicative double power law
with a smooth transition at the comoving cluster radius,

```text
Sigma_corr(R) = B_prj(R) Sigma_max(R),       Sigma_max = max(Sigma_1h, Sigma_2h)
B_prj(R)      = A (R/R0)^alpha [1 + (R/R0)^gamma]^((beta - alpha)/gamma) + 1
R0            = R_lambda(lob) (1 + z)         [comoving Mpc/h; R in the same units]
```

The same form fits the bias on DeltaSigma(R) with different values. Both
implementations evaluate `B(R, lob, z)` and read the four parameters from a
values-file section of the driving pipeline (default `costanzi_bprj`; pass
another section name to keep a Sigma and a DeltaSigma set in one pipeline):

```ini
[costanzi_bprj]
A = 0.10
alpha = 0.1
beta = -0.53
gamma = 4.1
```

DeltaSigma(R) best fit: `A = 0.12, alpha = 4.11, beta = 0.18, gamma = 1.82`
(`CostanziBprj.dsigma()` / `CostanziBprj_t::dsigma()`). Note: the arXiv
version of App. C quotes `alpha = 0.92` for Sigma; `0.1` above is the owner's
spec (2026-09-01) -- confirm against the published version before sampling.

```python
from systematics.costanzi_bprj.python.costanzi_bprj import CostanziBprj
bprj = CostanziBprj.from_datablock(block)       # or CostanziBprj.sigma()
sigma_corr = bprj(R, lob, zob) * sigma_max
```

```cpp
#include "pipelines/systematics/costanzi_bprj/cpp/costanzi_bprj_t.hh"
y3_cluster::CostanziBprj_t const bprj(sample);  // [costanzi_bprj] from the values file
double const sigma_corr = bprj(R, lob, zob) * sigma_max;
```

Consumer: `y3_buzzard/likelihood_cp.py` (`shear_max_section = shear1h2h_max`
+ `is_b_proj_costanzi26 = T`) multiplies the max-model shear theory by
`B(R, lob, z_bin)`, parameters from the values-file section `[costanzi_bprj]`.
Tests: `test/costanzi_bprj.test.{py,cc}`, `test/likelihood_cp.test.py`.
