// Miscentred one-halo shear — full (lambda_true, lnM, z) reference, CUDA.
// Physics lives in shear1h_full_ltmz_gpu_t.cuh; see that header for the
// integrand formula, configuration keys, and status notes.
#include "shear1h_full_ltmz_gpu_t.cuh"
#include "utils/cuda_module_macros.cuh"

DEFINE_COSMOSIS_CUDA_INTEGRATION_MODULE(Shear1hFullLtmzGpu)
