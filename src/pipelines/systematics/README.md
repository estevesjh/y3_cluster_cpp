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
