#include "catch2/catch.hpp"

#include "pipelines/des_y3/number_counts/cpp/3d/num_counts_3d_t.hh"

#include "models/dv_do_dz_t.hh"
#include "models/ez.hh"
#include "models/hmf_t.hh"
#include "models/mor_hod_t.hh"
#include "models/omega_z_des.hh"
#include "models/plob_ltr_emg_t.hh"
#include "models/richness_kernel_t.hh"
#include "utils/interp_1d.hh"
#include "utils/interp_2d.hh"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

using y3_cluster::MOR_HOD_t;
using y3_cluster::PlobLtrEMG_t;
using y3_cluster::RichnessKernel_t;
using y3_cluster::richness_zkernel;

namespace {
  constexpr double REL_TOL = 1.0e-3;

  // z-independent EMG parameters (constant Interp1D nodes) so K_i(ltr, z)
  // exercises richness_kernel_t.hh with the same closed-form math an
  // independent scipy.special.erf/erfc transcription reproduces below,
  // without needing a real PrjParams z-dependent table.
  cosmosis::DataBlock
  make_plob_config()
  {
    cosmosis::DataBlock cfg;
    std::vector<double> const z{0.10, 0.45, 0.80};
    auto const flat = [](double v) { return std::vector<double>{v, v, v}; };
    cfg.put_val("plob_ltr_params", "z", z);
    cfg.put_val("plob_ltr_params", "a_mu", flat(0.2));
    cfg.put_val("plob_ltr_params", "b_mu", flat(1.05));
    cfg.put_val("plob_ltr_params", "a_sig", flat(0.55));
    cfg.put_val("plob_ltr_params", "b_sig", flat(0.9));
    cfg.put_val("plob_ltr_params", "a_tau", flat(0.30));
    cfg.put_val("plob_ltr_params", "b_tau", flat(0.02));
    cfg.put_val("plob_ltr_params", "a_fprj", flat(1.2));
    cfg.put_val("plob_ltr_params", "b_fprj", flat(0.65));
    return cfg;
  }
}

TEST_CASE("NumCounts3d: K_i/K_j richness and photo-z kernels match an "
          "independent scipy erf/erfc transcription")
{
  auto cfg = make_plob_config();
  PlobLtrEMG_t const plob(cfg);
  RichnessKernel_t const k_i(20.0, 30.0);

  // Golden values from scipy.special.erf/erfc using the same closed-form
  // F_EMG/phi formulas documented in richness_kernel_t.hh (independent
  // Python re-derivation, not a call into this codebase's own replicas).
  CHECK(k_i(25.0, 0.35, plob) == Approx(2.427052281312100e-01).epsilon(REL_TOL));
  CHECK(k_i(60.0, 0.50, plob) == Approx(1.853491236150893e-05).epsilon(REL_TOL));
  CHECK(k_i(110.0, 0.65, plob) == Approx(1.258769684900066e-13).epsilon(REL_TOL));

  CHECK(richness_zkernel(0.30, 0.25, 0.45, 0.02) ==
        Approx(9.937903346741919e-01).epsilon(REL_TOL));
  CHECK(richness_zkernel(0.42, 0.25, 0.45, 0.02) ==
        Approx(9.331927987311421e-01).epsilon(REL_TOL));
  CHECK(richness_zkernel(0.55, 0.25, 0.45, 0.02) ==
        Approx(2.866515718680240e-07).epsilon(REL_TOL));
}

TEST_CASE("NumCounts3d: MOR_HOD_t shifted-Poisson matches an "
          "independent Python (scipy.special.gammaln) transcription")
{
  MOR_HOD_t const mor(/*log10_Mmin=*/13.8, /*log10_M1=*/14.5, /*alpha=*/1.1,
                      /*epsilon=*/-0.2, /*sigma_lambda=*/0.35);
  double const lnM = std::log(std::pow(10.0, 14.2));

  CHECK(mor(5.0, lnM, 0.3) == Approx(5.411587466357029e-03).epsilon(REL_TOL));
  CHECK(mor(12.0, lnM, 0.5) == Approx(5.639899981958428e-12).epsilon(REL_TOL));
  CHECK(mor(25.0, lnM, 0.6) == Approx(7.820106802009195e-34).epsilon(REL_TOL));
}

TEST_CASE("NumCounts3d: operator() equals the documented product of "
          "HMF * dV/dOdz * Omega(z) * K_j * K_i * P_HOD")
{
  // Config: one richness/photo-z bin.
  cosmosis::DataBlock cfg;
  cfg.put_val("NumCounts3d", "lam_min", std::vector<double>{20.0});
  cfg.put_val("NumCounts3d", "lam_max", std::vector<double>{30.0});
  cfg.put_val("NumCounts3d", "zob_min", std::vector<double>{0.25});
  cfg.put_val("NumCounts3d", "zob_max", std::vector<double>{0.45});
  cfg.put_val("NumCounts3d", "sigma_z", std::vector<double>{0.02});
  NumCounts3d model(cfg);

  // Sample: a small synthetic HMF grid (3x3), a synthetic d_a(z) table, and
  // the same plob/HOD parameters used in the two tests above.
  cosmosis::DataBlock sample;
  sample.put_val<double>("cosmological_parameters", "omega_M", 0.3);
  sample.put_val<double>("cosmological_parameters", "omega_nu", 0.0);
  sample.put_val<double>("cosmological_parameters", "omega_k", 0.0);
  sample.put_val<double>("cosmological_parameters", "omega_lambda", 0.7);
  sample.put_val<double>("cosmological_parameters", "h0", 0.7);

  // HMF: mass_function/{m_h, z, dndlnmh} with m_h chosen so that
  // ln(m_h * (omega_m - omega_nu)) equals the {30, 32, 34} query grid used
  // by the Python fixture, i.e. m_h = exp({30,32,34}) / 0.3.
  std::vector<double> const lnm_query{30.0, 32.0, 34.0};
  std::vector<double> m_h;
  for (double lnm : lnm_query) m_h.push_back(std::exp(lnm) / 0.3);
  std::vector<double> const z_grid{0.2, 0.4, 0.6};
  auto const hmf_raw = [](double lnM, double z) {
    return std::exp(-(lnM - 32.0) * (lnM - 32.0) / 8.0) * (1.0 + z);
  };
  // Row-major flat (z slowest, lnM fastest), matching Interp2D's ndarray
  // ctor convention (extents = {ny, nx} = {z.size(), lnm.size()}).
  std::vector<double> dndlnmh_flat;
  dndlnmh_flat.reserve(z_grid.size() * lnm_query.size());
  for (double z : z_grid)
    for (double lnm : lnm_query) dndlnmh_flat.push_back(hmf_raw(lnm, z));
  cosmosis::ndarray<double> const dndlnmh(
    dndlnmh_flat, std::vector<std::size_t>{z_grid.size(), lnm_query.size()});
  sample.put_val("mass_function", "m_h", m_h);
  sample.put_val("mass_function", "z", z_grid);
  sample.put_val("mass_function", "dndlnmh", dndlnmh);
  sample.put_val<double>("cluster_abundance", "hmf_s", 0.08);
  sample.put_val<double>("cluster_abundance", "hmf_q", 0.02);

  std::vector<double> const z_da{0.0, 0.3, 0.6, 0.9};
  std::vector<double> const d_a{0.0, 900.0, 1500.0, 1800.0};
  sample.put_val("distances", "z", z_da);
  sample.put_val("distances", "d_a", d_a);

  sample.put_val<double>("cluster_mor", "log10_Mmin", 13.8);
  sample.put_val<double>("cluster_mor", "log10_M1", 14.5);
  sample.put_val<double>("cluster_mor", "alpha", 1.1);
  sample.put_val<double>("cluster_mor", "epsilon", -0.2);
  sample.put_val<double>("cluster_mor", "sigma_lambda", 0.35);

  std::vector<double> const z_plob{0.10, 0.45, 0.80};
  auto const flat = [](double v) { return std::vector<double>{v, v, v}; };
  sample.put_val("plob_ltr_params", "z", z_plob);
  sample.put_val("plob_ltr_params", "a_mu", flat(0.2));
  sample.put_val("plob_ltr_params", "b_mu", flat(1.05));
  sample.put_val("plob_ltr_params", "a_sig", flat(0.55));
  sample.put_val("plob_ltr_params", "b_sig", flat(0.9));
  sample.put_val("plob_ltr_params", "a_tau", flat(0.30));
  sample.put_val("plob_ltr_params", "b_tau", flat(0.02));
  sample.put_val("plob_ltr_params", "a_fprj", flat(1.2));
  sample.put_val("plob_ltr_params", "b_fprj", flat(0.65));

  model.set_sample(sample);
  model.set_grid_point({0});

  double const lt = 8.0, zt = 0.40, lnM = 33.1;
  double const got = model(lt, zt, lnM);

  // Independently-constructed sub-models built from the same raw values
  // (not read back out of `model`, which keeps them private) recompute the
  // documented product; a wiring bug (missing/duplicated/reordered term)
  // in NumCounts3d::operator() would show up as a mismatch here even
  // without an external Python reference.
  // matrix<M,N>[i][j] pairs xs[i] (lnM) with ys[j] (z) -- see
  // Interp2D's array-based ctor in utils/interp_2d.hh.
  std::array<double, 3> const lnm_axis{30.0, 32.0, 34.0};
  std::array<double, 3> const z_axis{0.2, 0.4, 0.6};
  y3_cluster::Interp2D::matrix<3, 3> const hmf_grid{
    {{hmf_raw(30.0, 0.2), hmf_raw(30.0, 0.4), hmf_raw(30.0, 0.6)},
     {hmf_raw(32.0, 0.2), hmf_raw(32.0, 0.4), hmf_raw(32.0, 0.6)},
     {hmf_raw(34.0, 0.2), hmf_raw(34.0, 0.4), hmf_raw(34.0, 0.6)}}};
  y3_cluster::Interp2D const hmf_interp(lnm_axis, z_axis, hmf_grid);
  y3_cluster::HMF_t const hmf(hmf_interp, 0.08, 0.02);
  y3_cluster::Interp1D const da_interp(z_da, d_a);
  y3_cluster::DV_DO_DZ_t const dv(da_interp, y3_cluster::EZ(0.3, 0.7, 0.0),
                                  0.7);
  y3_cluster::OMEGA_Z_DES const omega;
  MOR_HOD_t const mor(13.8, 14.5, 1.1, -0.2, 0.35);
  auto plob_cfg = make_plob_config();
  PlobLtrEMG_t const plob(plob_cfg);
  RichnessKernel_t const k_i(20.0, 30.0);

  double const expected = hmf(lnM, zt) * dv(zt) * omega(zt) *
                          richness_zkernel(zt, 0.25, 0.45, 0.02) *
                          k_i(lt, zt, plob) * mor(lt, lnM, zt);

  CHECK(got == Approx(expected).epsilon(1e-12));
  // Also pin against the independent Python re-derivation of the whole
  // chain (scipy RegularGridInterpolator-equivalent bilinear + the erf/erfc
  // kernels above), tying the composition to physics, not just self-
  // consistency.
  CHECK(got == Approx(4.542270107063506e+02).epsilon(REL_TOL));
}

namespace {
  // A second, fully-independent fixture (Einstein-de Sitter cosmology, a
  // 2x2 (lnM,z) HMF grid that is exactly bilinear, linear d_a(z)) chosen
  // so every constituent piece has zero interpolation error, used below to
  // test bin-selection/support-boundary invariants and a second closed-form
  // pin distinct from the fixture above.
  cosmosis::DataBlock
  make_invariants_sample()
  {
    cosmosis::DataBlock sample;

    sample.put_val<double>("cosmological_parameters", "omega_m", 1.0);
    sample.put_val<double>("cosmological_parameters", "omega_nu", 0.0);
    sample.put_val<double>("cosmological_parameters", "omega_lambda", 0.0);
    sample.put_val<double>("cosmological_parameters", "omega_k", 0.0);
    sample.put_val<double>("cosmological_parameters", "h0", 0.65);

    std::vector<double> const lnm_nodes{32.0, 34.0};
    std::vector<double> m_h;
    for (double lnm : lnm_nodes) m_h.push_back(std::exp(lnm));
    std::vector<double> const z_nodes{0.3, 0.5};
    auto const dndlnmh_fn = [](double lnM, double z) {
      return 5.0 - 0.1 * lnM + 2.0 * z + 0.05 * lnM * z;
    };
    std::vector<double> dndlnmh_flat;
    dndlnmh_flat.reserve(z_nodes.size() * lnm_nodes.size());
    for (double z : z_nodes)
      for (double lnm : lnm_nodes) dndlnmh_flat.push_back(dndlnmh_fn(lnm, z));
    cosmosis::ndarray<double> const dndlnmh(
      dndlnmh_flat,
      std::vector<std::size_t>{z_nodes.size(), lnm_nodes.size()});
    sample.put_val("mass_function", "m_h", m_h);
    sample.put_val("mass_function", "z", z_nodes);
    sample.put_val("mass_function", "dndlnmh", dndlnmh);
    sample.put_val<double>("cluster_abundance", "hmf_s", 0.05);
    sample.put_val<double>("cluster_abundance", "hmf_q", 0.95);

    std::vector<double> const z_da{0.0, 1.0};
    std::vector<double> const d_a{0.0, 1500.0};
    sample.put_val("distances", "z", z_da);
    sample.put_val("distances", "d_a", d_a);

    sample.put_val<double>("cluster_mor", "log10_Mmin", 13.8);
    sample.put_val<double>("cluster_mor", "log10_M1", 14.5);
    sample.put_val<double>("cluster_mor", "alpha", 1.1);
    sample.put_val<double>("cluster_mor", "epsilon", -0.2);
    sample.put_val<double>("cluster_mor", "sigma_lambda", 0.35);

    std::vector<double> const z_plob{0.10, 0.45, 0.80};
    auto const flat = [](double v) { return std::vector<double>{v, v, v}; };
    sample.put_val("plob_ltr_params", "z", z_plob);
    sample.put_val("plob_ltr_params", "a_mu", flat(0.35));
    sample.put_val("plob_ltr_params", "b_mu", flat(0.95));
    sample.put_val("plob_ltr_params", "a_sig", flat(0.5));
    sample.put_val("plob_ltr_params", "b_sig", flat(1.1));
    sample.put_val("plob_ltr_params", "a_tau", flat(0.28));
    sample.put_val("plob_ltr_params", "b_tau", flat(0.015));
    sample.put_val("plob_ltr_params", "a_fprj", flat(1.4));
    sample.put_val("plob_ltr_params", "b_fprj", flat(0.55));

    return sample;
  }
}

TEST_CASE("NumCounts3d: bin selection and support-boundary invariants")
{
  cosmosis::DataBlock cfg;
  cfg.put_val("NumCounts3d", "lam_min", std::vector<double>{15.0, 50.0});
  cfg.put_val("NumCounts3d", "lam_max", std::vector<double>{25.0, 70.0});
  cfg.put_val("NumCounts3d", "zob_min", std::vector<double>{0.15, 0.40});
  cfg.put_val("NumCounts3d", "zob_max", std::vector<double>{0.30, 0.55});
  cfg.put_val("NumCounts3d", "sigma_z", std::vector<double>{0.025, 0.03});
  NumCounts3d model(cfg);

  auto sample = make_invariants_sample();
  model.set_sample(sample);

  double const lt = 20.0, zt = 0.28, lnM = 33.0;

  model.set_grid_point({0});
  double const v0 = model(lt, zt, lnM);
  CHECK(std::isfinite(v0));
  CHECK(v0 > 0.0);

  // Same evaluation point, different bin: lt = 20 sits inside bin 0's
  // richness range [15, 25] but far below bin 1's [50, 70], so K_i for
  // bin 1 collapses the whole product to (near) zero. A bug that ignores
  // bin_index, or caches the first bin's kernel across set_grid_point
  // calls, would leave v1 close to v0 instead.
  model.set_grid_point({1});
  double const v1 = model(lt, zt, lnM);
  CHECK(std::isfinite(v1));
  CHECK(v1 < v0 * 1.0e-3);

  // Out-of-range bin indices must throw, per the documented contract in
  // num_counts_3d_t.hh::set_grid_point.
  CHECK_THROWS_AS(model.set_grid_point({2}), std::out_of_range);
  CHECK_THROWS_AS(model.set_grid_point({-1}), std::out_of_range);

  // Physical support boundary: lambda_true < lambda_central (= 1 once
  // M >= Mmin) is unphysical for MOR_HOD_t (lambda_sat would be
  // negative), which returns exactly 0 -- propagating to an exact 0 for
  // the whole product regardless of the other four terms.
  model.set_grid_point({0});
  CHECK(model(0.5, zt, lnM) == 0.0);
}

TEST_CASE("NumCounts3d: operator() matches a second, fully-independent "
          "closed-form fixture (Einstein-de Sitter, exact-bilinear HMF)")
{
  cosmosis::DataBlock cfg;
  cfg.put_val("NumCounts3d", "lam_min", std::vector<double>{2.0});
  cfg.put_val("NumCounts3d", "lam_max", std::vector<double>{4.5});
  cfg.put_val("NumCounts3d", "zob_min", std::vector<double>{0.2});
  cfg.put_val("NumCounts3d", "zob_max", std::vector<double>{0.5});
  cfg.put_val("NumCounts3d", "sigma_z", std::vector<double>{0.05});
  NumCounts3d model(cfg);

  auto sample = make_invariants_sample();
  model.set_sample(sample);
  model.set_grid_point({0});

  double const lt = 3.0, zt = 0.4, lnM = 33.0;
  double const got = model(lt, zt, lnM);

  y3_cluster::HMF_t const hmf(
    y3_cluster::Interp2D({32.0, 34.0}, {0.3, 0.5},
                        std::vector<double>{5.0 - 0.1 * 32.0 + 2.0 * 0.3 +
                                              0.05 * 32.0 * 0.3,
                                            5.0 - 0.1 * 34.0 + 2.0 * 0.3 +
                                              0.05 * 34.0 * 0.3,
                                            5.0 - 0.1 * 32.0 + 2.0 * 0.5 +
                                              0.05 * 32.0 * 0.5,
                                            5.0 - 0.1 * 34.0 + 2.0 * 0.5 +
                                              0.05 * 34.0 * 0.5}),
    0.05, 0.95);
  y3_cluster::Interp1D const da_interp(std::vector<double>{0.0, 1.0},
                                       std::vector<double>{0.0, 1500.0});
  y3_cluster::DV_DO_DZ_t const dv(da_interp, y3_cluster::EZ(1.0, 0.0, 0.0),
                                  0.65);
  y3_cluster::OMEGA_Z_DES const omega;
  MOR_HOD_t const mor(13.8, 14.5, 1.1, -0.2, 0.35);
  RichnessKernel_t const k_i(2.0, 4.5);
  cosmosis::DataBlock plob_cfg;
  {
    std::vector<double> const z_plob{0.10, 0.45, 0.80};
    auto const flat = [](double v) { return std::vector<double>{v, v, v}; };
    plob_cfg.put_val("plob_ltr_params", "z", z_plob);
    plob_cfg.put_val("plob_ltr_params", "a_mu", flat(0.35));
    plob_cfg.put_val("plob_ltr_params", "b_mu", flat(0.95));
    plob_cfg.put_val("plob_ltr_params", "a_sig", flat(0.5));
    plob_cfg.put_val("plob_ltr_params", "b_sig", flat(1.1));
    plob_cfg.put_val("plob_ltr_params", "a_tau", flat(0.28));
    plob_cfg.put_val("plob_ltr_params", "b_tau", flat(0.015));
    plob_cfg.put_val("plob_ltr_params", "a_fprj", flat(1.4));
    plob_cfg.put_val("plob_ltr_params", "b_fprj", flat(0.55));
  }
  PlobLtrEMG_t const plob(plob_cfg);

  double const expected = hmf(lnM, zt) * dv(zt) * omega(zt) *
                          richness_zkernel(zt, 0.2, 0.5, 0.05) *
                          k_i(lt, zt, plob) * mor(lt, lnM, zt);
  CHECK(got == Approx(expected).epsilon(1.0e-9));

  // Independent end-to-end reference: closed-form HMF (the 2x2 grid is
  // exactly bilinear, so GSL's bilinear interpolation reproduces it with
  // zero interpolation error), closed-form dV/dOdz (Einstein-de Sitter
  // EZ(z) = (1+z)^1.5, linear d_a(z)), and the same richness_kernel_t.hh/
  // mor_hod_t.hh formulas re-derived independently in Python
  // (scipy.special.erf/erfc/gammaln) rather than called back into this
  // codebase.
  CHECK(got == Approx(58314101.196420856).epsilon(REL_TOL));
}
