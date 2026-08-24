// Full (lambda_true, lnM, z) reference number counts — `full_ltmz`, CUDA.
// Physics lives in num_counts_full_ltmz_gpu_t.cuh; see that header for
// the integrand formula, configuration keys, and status notes.
#include "num_counts_full_ltmz_gpu_t.cuh"
#include "utils/cuda_module_macros.cuh"

DEFINE_COSMOSIS_CUDA_INTEGRATION_MODULE(NumCountsFullLtmzGpu)
