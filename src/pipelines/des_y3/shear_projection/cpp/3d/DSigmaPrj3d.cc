// Projection shear -- full (theta_mis, z, lnM) reference, C++/cubacpp Cuhre.
// Physics lives in dsigma_prj_3d_t.hh; see that header for the
// integrand formula, configuration keys, and status notes.
#include "dsigma_prj_3d_t.hh"
#include "utils/module_macros.hh"

DEFINE_COSMOSIS_SCALAR_INTEGRATION_MODULE(DSigmaPrj3d)
