// Traditional 1h+2h max-model shear — full (lambda_true, lnM, z)
// reference, CUDA. Physics lives in shear1h2h_max_3d_gpu_t.cuh
#include "shear1h2h_max_3d_gpu_t.cuh"
#include "utils/cuda_module_macros.cuh"

DEFINE_COSMOSIS_CUDA_INTEGRATION_MODULE(Shear1h2hMax3dGpu)
