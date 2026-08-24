// Projection shear, frozen-physics fast path — GPU adaptation.
// Physics lives in shear_prj_frozen_gpu_t.cuh; see that header for the
// algorithm, configuration keys, and status notes.
#include "shear_prj_frozen_gpu_t.cuh"
#include "utils/module_macros.hh"

DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE(ShearPrjFrozenGpu)
