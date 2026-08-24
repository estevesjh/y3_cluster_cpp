// Traditional 1h+2h shear via the max model — C++ backend.
// Physics lives in shear1h2h_max_t.hh; see that header for the max-model
// composition, configuration keys, and status notes.
#include "shear1h2h_max_t.hh"
#include "utils/module_macros.hh"

DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE(Shear1h2hMax)
