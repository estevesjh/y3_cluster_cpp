// Adaptive (Cuhre/Vegas inner z+lnM) reference path for Sigma_prj /
// DSigma_prj / gamma_t^prj.  Outer theta is log-GL on segments split at
// feature breakpoints -- shared with the fixed-GL ShearPrjEvaluator.
// Writes sigma_prj_cuhre / dsigma_prj_cuhre / shear_prj_cuhre.
#include "models/sigma_prj_t.hh"
#include "utils/module_macros.hh"

using ShearPrjCuhre = y3_cluster::ShearPrjCuhre;

DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE(ShearPrjCuhre)
