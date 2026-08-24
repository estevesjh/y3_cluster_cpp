# Partially adaptive projection integration (`2d`)

This strategy has two adaptive dimensions: the outer angular integral is a
fixed log-Gauss--Legendre grid split at the per-radius feature breakpoints
(shared with the `0d` fixed-grid evaluators), and adaptive Cuhre or Vegas
handles only the inner two-dimensional `(z, lnM)` integral. The feature-split
angular treatment is exactly what protects this backend from the missed-cusp
failure mode of the fully-coupled [`../3d/`](../3d/README.md) diagnostic,
while the adaptive inner integral makes it an independent convergence check
on the region-split fixed-GL redshift/mass grids of
[`../0d/`](../0d/README.md).

## Language implementation

| Language | Algorithm and source file | Status |
| --- | --- | --- |
| Python | No module in this namespace. The Python exact-z `0d` implementation is the readable reference. | Not implemented |
| C++ | `cpp/ShearPrjCuhre.cc` (moved here from `src/modules/sigma_prj_cpu/`; one-line instantiation over the immutable `models/sigma_prj_t.hh` core, `y3_cluster::ShearPrjCuhre`). | Implemented adaptive comparison backend |
| CUDA | No implementation. The fully-coupled PAGANI diagnostic lives in [`../3d/`](../3d/README.md). | Not implemented |

Output sections are unchanged by the move: `sigma_prj_cuhre` /
`dsigma_prj_cuhre` / `shear_prj_cuhre`. The compiled module lands in
`release-build/src/modules/des_y3_shear_prj_2d_cpp/ShearPrjCuhre.so`.

## Role

Used for head-to-head convergence diffs against the `0d` fixed-GL evaluators
(`ShearPrjEvaluator` and kin) and the Python exact-z reference. It is a
diagnostic/comparison backend, not a production entry point. The fixed-grid
and GSL alternatives are summarized in the
[projection numerical-method map](../README.md#what-the-numerical-methods-actually-integrate).
