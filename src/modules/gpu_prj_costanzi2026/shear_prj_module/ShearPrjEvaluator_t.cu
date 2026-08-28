
// projection-shear module wrapper
// It directly evaluates the projection shear following sigma_prj_t.hh / ShearPrjEvaluator.cc.
//
// shear_prj/vals = DSigmaTotalWeight = rnd + cl in one pass, i.e. the full
// [1 + b(M,z)*b_sel(theta)*xi_NL] * DSigma_mis integrand -- includes the
// selection-bias-weighted clustering term (from bSelMargGPU's B_small/
// B_large), not just the uncorrelated background piece. Was
// DSigmaRndWeight (background only) until this fix.

#include "sigma_prj_gpu_t.cuh"
#include "utils/cuda_module_macros.cuh"

using ShearPrjEvaluator = y3_cuda::SigmaPrjGPU<y3_cuda::DSigmaTotalWeight>;

DEFINE_COSMOSIS_CUDA_INTEGRATION_MODULE(ShearPrjEvaluator);