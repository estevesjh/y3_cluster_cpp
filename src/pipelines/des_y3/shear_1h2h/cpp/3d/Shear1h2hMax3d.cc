// Traditional 1h+2h max-model shear — full (lambda_true, lnM, z)
// reference, C++. Physics lives in shear1h2h_max_3d_t.hh; see
// that header for the integrand formula, configuration keys, and
// status notes.
#include "shear1h2h_max_3d_t.hh"
#include "utils/module_macros.hh"

DEFINE_COSMOSIS_SCALAR_INTEGRATION_MODULE(Shear1h2hMax3d)
