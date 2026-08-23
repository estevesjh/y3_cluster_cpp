// Miscentred one-halo shear via the radial_series strategy (offline
// U_ell tables + population moments; see shear1h_radial_series_t.hh and
// ../python/README.md for the contract and validation record).
//
// Reference/candidate backend under src/pipelines/des_y3; the
// production stage remains Shear1hMisSel.so.
#include "pipelines/des_y3/observables/shear_1h2h/radial_series/cpp/shear1h_radial_series_t.hh"
#include "utils/module_macros.hh"

using Shear1hRadialSeries = y3_cluster::des_y3::Shear1hRadialSeries;

DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE(Shear1hRadialSeries)
