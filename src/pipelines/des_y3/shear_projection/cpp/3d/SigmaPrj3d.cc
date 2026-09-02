// Projection surface density -- full (theta_mis, z, lnM) reference,
// C++/cubacpp Cuhre. Physics lives in sigma_prj_3d_t.hh; see that
// header (and dsigma_prj_3d_t.hh, which it mirrors) for the integrand
// formula, configuration keys, and status notes.
#include "sigma_prj_3d_t.hh"
#include "utils/module_macros.hh"

DEFINE_COSMOSIS_SCALAR_INTEGRATION_MODULE(SigmaPrj3d)
