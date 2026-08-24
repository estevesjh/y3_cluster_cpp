// Shared lensing helpers for the pipeline C++/CUDA backends.
//
// Pipeline-local replacements for the handful of helpers the des_y3
// lensing drivers used to pull from the production module header
// modules/num_counts_sel/lensing_weights.hh — pipeline code must not
// include production module headers, so these live here in the shared
// layer (the C++ sibling of shared/datablock_models.py, which carries
// the same helpers for the Python backends). Deliberately NOT carried
// over: read_mis_param and the F_MIS/TAU_MIS defaults — f_mis and
// tau_mis are required datablock values (miscentering/f_mis,
// miscentering/tau_mis) read with view<double>, which throws if the
// section is missing.
//
// Sigma_crit_inv(z_lens) is the source-bin-weighted 1-D table published
// by src/modules/average_sigma_crit_inv/average_sigma_crit_inv.py (axes
// "zlense" / "sci_average"). The lensing profile tables come from the
// "haloModel" section published by the cosmology halo-model stage
// (src/pipelines/cosmology/halo_model.py, driven through
// y3_buzzard/halo_model_cosmosis.py).
#ifndef Y3_CLUSTER_CPP_PIPELINES_SHARED_LENSING_HELPERS_HH
#define Y3_CLUSTER_CPP_PIPELINES_SHARED_LENSING_HELPERS_HH

#include "cosmosis/datablock/datablock.hh"
#include "utils/datablock_reader.hh"
#include "utils/interp_1d.hh"
#include "utils/make_interp_1d.hh"

#include <cmath>
#include <vector>

namespace y3_pipelines {

  // Sigma_crit_inv^{-1}(z_lens) 1-D table.
  inline y3_cluster::Interp1D
  load_sigma_crit_inv(cosmosis::DataBlock& s)
  {
    return y3_cluster::make_Interp1D(s, "average_sigma_crit_inv",
                                     "zlense", "sci_average");
  }

  // R_lambda(lambda^ob) = (lambda^ob / 100)^0.2  [h^-1 Mpc]
  inline double
  R_lambda(double lob)
  {
    return std::pow(lob / 100.0, 0.2);
  }

  // Default DES-Y3 richness-bin centres (edges [20, 30, 45, 60, 200]),
  // the same default shared/datablock_models.py::DEFAULT_LOB_CENTERS
  // gives the Python backends.
  inline std::vector<double>
  default_lob_centers()
  {
    return {25.0, 37.5, 52.5, 130.0};
  }

  // Load lob_centers from the module section if present, else default.
  inline std::vector<double>
  read_lob_centers(cosmosis::DataBlock& s, char const* module_section)
  {
    if (s.has_val(module_section, "lob_centers"))
      return get_vector_double(s, module_section, "lob_centers");
    return default_lob_centers();
  }

}  // namespace y3_pipelines

#endif
