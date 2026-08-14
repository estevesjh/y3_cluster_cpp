// N_i[1] — number counts in (lambda_ob, z_ob) bin using the closed-form
// Costanzi-Y3 richness selection function.
//
// Fixed-GL evaluator (src/models/n_operator_sel_gl_t.hh): the 2-D
// (zt, lnM) quadrature is done once per bin on fixed Gauss-Legendre
// nodes in set_sample, replacing the per-bin adaptive Cuhre integral of
// the retired NOperatorSelScalar<WeightOne> version.  Besides the mean
// speed-up (0.107 s -> 0.021 s), the cost is now deterministic: over
// ~1e6 MCMC realisations the Cuhre version ranged up to 0.98 s (9x its
// mean) at hard corners of parameter space.  Same module label, grid
// semantics (bin_index wall) and output (numcountssel/vals); the old
// algorithm/eps_rel/eps_abs/max_eval/use_cartesian_product ini knobs
// are ignored.
#include "models/n_operator_sel_gl_t.hh"
#include "utils/module_macros.hh"

using NumCountsSel = y3_cluster::NumCountsSelGL;

DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE(NumCountsSel)
