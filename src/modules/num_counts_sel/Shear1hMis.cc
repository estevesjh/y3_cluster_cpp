// N_i[gamma_t^1h, full(R)] -- 1-halo tangential shear with miscentering:
//
//   gamma_t^1h, full(R, M, z) = DSigma_cl(R, M, z) * Sigma_crit^-1(z_lens)
//
// where DSigma_cl is the (centred + miscentered) mixture defined in
// DSigma1hMis.cc.  Ratio Shear1hMisSel / NumCountsSel gives the
// bin-averaged <gamma_t^1h, full>(R) ready to be summed with the
// gamma_t^prj projection observable in the likelihood.
//
// Fixed-GL evaluator (src/models/n_operator_sel_gl_t.hh): the profile
// is z-free, so the z-marginalised mass weight is built once per bin in
// set_sample and every radius is served from the cache, replacing the
// per-(bin, R) adaptive Cuhre integrals of the retired
// NOperatorSelRadial<Shear1hMisWeight> version (~16x faster; see
// docs/shear1h_radial_factorization.tex).  Same module label, grid
// semantics (bin_index x r_perp cartesian product) and output
// (shear1hmissel/vals).  NOTE: bins 4-11 now use their own richness
// bin's R_lambda (bin_index % 4) instead of silently reusing bin 3's --
// a deliberate fix, changing those bins by up to ~2%.
#include "models/n_operator_sel_gl_t.hh"
#include "utils/module_macros.hh"

using Shear1hMisSel = y3_cluster::Shear1hMisSelGL;

DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE(Shear1hMisSel)
