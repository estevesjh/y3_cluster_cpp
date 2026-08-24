// Cluster number counts via the fast_mass strategy — C++ backend.
//
// By identity: the fixed-GL quadrature (src/models/n_operator_sel_gl_t.hh,
// SelGLCore) IS the fast_mass algorithm for number counts — there is no
// separate inline-selection variant here (see the reverted
// NumCountsFastMass/sel_function_inline_t.hh attempt: duplicating
// sel_function.py's tensor per-driver cost ~2x more than sharing it).
// This class is the des_y3-namespaced wrapper around the same
// NumCountsSelGL body, with its own module label and output section so
// it can co-run alongside production NumCountsSel.so for comparison
// (DataBlock sections do not overwrite, so the two must never target
// the same section).
//
// Options: bin_index wall (default = every configured bin), required
// zt_low/zt_high/lnm_low/lnm_high, n_lnm (96), n_z (64) — identical
// names/semantics to src/pipelines/des_y3/README.md's "by identity"
// reference-choice entry for this observable.
#ifndef Y3_CLUSTER_CPP_NUM_COUNTS_FAST_MASS_T_HH
#define Y3_CLUSTER_CPP_NUM_COUNTS_FAST_MASS_T_HH

#include "models/n_operator_sel_gl_t.hh"
#include "utils/make_grid_points.hh"

#include <array>
#include <stdexcept>

class NumCountsFastMassCpp {
public:
  using grid_t = y3_cluster::grid_t<1>;
  using grid_point_t = grid_t::value_type;
  static constexpr std::size_t n_outputs = 1;

  explicit NumCountsFastMassCpp(cosmosis::DataBlock& cfg)
    : core_(cfg, module_label())
  {}

  void
  set_sample(cosmosis::DataBlock& s)
  {
    core_.build_weights(s, /*include_sci=*/false);
  }

  std::array<double, n_outputs>
  evaluate(grid_point_t const& pt) const
  {
    int const b = static_cast<int>(pt[0]);
    if (b < 0 || static_cast<std::size_t>(b) >= core_.n_bins())
      throw std::out_of_range(
        "NumCountsFastMassCpp: bin_index outside sel_function/S_stack range");
    return {core_.norm(b)};
  }

  static char const* module_label() { return "NumCountsFastMass"; }

  static std::array<char const*, n_outputs>
  output_sections()
  {
    return {"numcounts_fast_mass"};
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::make_grid_points_wall_of_numbers(
      cfg, module_label(), "bin_index");
  }

private:
  y3_cluster::nosel_gl_detail::SelGLCore core_;
};

#endif
