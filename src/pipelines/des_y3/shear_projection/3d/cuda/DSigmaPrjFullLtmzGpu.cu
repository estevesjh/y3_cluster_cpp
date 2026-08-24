// Projection shear — full (theta, z, lnM)-resolved reference, CUDA/PAGANI.
// Physics lives in dsigma_prj_full_ltmz_gpu_t.cuh; see that header for
// the integrand formula, configuration keys, and status notes.
#include "dsigma_prj_full_ltmz_gpu_t.cuh"
#include "utils/cuda_module_macros.cuh"

DEFINE_COSMOSIS_CUDA_INTEGRATION_MODULE(DSigmaPrjFullLtmzGpu)
