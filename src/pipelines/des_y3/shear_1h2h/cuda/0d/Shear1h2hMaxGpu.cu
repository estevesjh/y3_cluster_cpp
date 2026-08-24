// Traditional 1h+2h shear via the max model — GPU adaptation.
// Physics lives in shear1h2h_max_gpu_t.cuh; see that header for the
// max-model composition, configuration keys, and status notes.
#include "shear1h2h_max_gpu_t.cuh"
#include "utils/module_macros.hh"

DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE(Shear1h2hMaxGpu)
