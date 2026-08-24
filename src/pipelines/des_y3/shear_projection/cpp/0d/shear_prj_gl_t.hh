// Projection shear via the S_ij-tabulated fixed-GL path (formerly fast_mass) — C++ backend.
//
// Thin des_y3 driver over the immutable exact core
// (sp_detail::ShearPrjCore, systematics/shear_prj/cpp/sigma_prj_t.hh): the exact
// redshift contraction with no frozen-physics approximation — the same
// computation as the Python backend (../python/shear_prj_gl.py,
// which was validated against this core's evaluators to 1.6e-11) and
// the same algorithm the dsigma_prj / shear_prj diagnostic evaluators
// expose. This driver publishes BOTH observables from ONE core (the
// existing evaluators each build their own), under the namespace
// sections, so the two backends are drop-in interchangeable:
//   dsigma_prj_gl/{vals,rnd,cl}   Msun/(h pc^2)
//   shear_prj_gl/{vals,rnd,cl}    dimensionless
// Ini section: ShearPrjGl (ShearPrjCore wall + knobs).
// Status: reference backend; production remains ShearPrjFrozenPhysics.
#ifndef Y3_CLUSTER_CPP_SHEAR_PRJ_GL_T_HH
#define Y3_CLUSTER_CPP_SHEAR_PRJ_GL_T_HH

#include "pipelines/systematics/shear_prj/cpp/sigma_prj_t.hh"

#include <array>

class ShearPrjGl {
public:
  using grid_t = y3_cluster::sp_detail::ShearPrjCore::grid_t;
  using grid_point_t = y3_cluster::sp_detail::ShearPrjCore::grid_point_t;
  static constexpr std::size_t n_outputs = 6;

  explicit ShearPrjGl(cosmosis::DataBlock& cfg)
    : core_(cfg, module_label())
  {}

  void set_sample(cosmosis::DataBlock& s) { core_.set_sample(s); }

  std::array<double, n_outputs>
  evaluate(grid_point_t const& pt) const
  {
    auto const ds = core_.dsigma_prj(pt);
    auto const sh = core_.shear_prj(pt);
    return {ds[0], ds[1], ds[2], sh[0], sh[1], sh[2]};
  }

  static char const* module_label() { return "ShearPrjGl"; }

  static std::array<char const*, n_outputs>
  output_sections()
  {
    return {"dsigma_prj_gl", "dsigma_prj_gl",
            "dsigma_prj_gl", "shear_prj_gl",
            "shear_prj_gl", "shear_prj_gl"};
  }

  static std::array<char const*, n_outputs>
  output_names()
  {
    return {"vals", "rnd", "cl", "vals", "rnd", "cl"};
  }

  static grid_t
  make_grid_points(cosmosis::DataBlock& cfg)
  {
    return y3_cluster::sp_detail::ShearPrjCore::make_grid_points(
      cfg, module_label());
  }

private:
  y3_cluster::sp_detail::ShearPrjCore core_;
};

#endif
