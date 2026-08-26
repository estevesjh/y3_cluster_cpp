// Unit tests for the S_ij-tabulated fixed-GL number-counts C++ backend
// (src/pipelines/des_y3/number_counts/cpp/0d/num_counts_sij_gl_t.hh, module
// NumCountsSijGl.cc) and the pipeline-owned weight core it is a wrapper
// around, y3_pipelines::SelGlWeights (src/pipelines/shared/sel_gl_weights.hh).
//
// This is the reference pipeline's production 0d counts backend and it had
// ZERO unit-test coverage: no Catch2 target instantiated the header, and
// numcounts_cross_backend.test.py exercises the Python replica and the
// adaptive 3d backend, never this class.
//
// What is pinned here, on a small self-contained synthetic DataBlock
// (two richness bins, two knots per table axis -- enough because every
// table below is either constant or linear along the one axis the formula
// varies, so GSL's bilinear/linear interpolators reproduce it exactly at
// any in-range query, exactly as in test/shear1h_gl.test.cc):
//
//   1. NumCountsSijGl::evaluate() equals an independently assembled
//      reference that recombines the SAME separately-tested leaf models
//      (HMF_t, DV_DO_DZ_t, OMEGA_Z_DES, SelFunction_t) per the
//      N_ij = int dlnM int dz n dV/dOmegadz Omega S_ij formula documented
//      at the top of sel_gl_weights.hh.
//   2. The Sigma_crit^-1 contract: counts call build_weights with
//      include_sci = false, so the result must NOT change when
//      average_sigma_crit_inv changes -- the one structural difference
//      between the counts and shear uses of the shared core. (The shear
//      side of the same core is pinned by shear1h_gl.test.cc.)
//   3. The HMF mass-shift contract (CLAUDE.md "Unit conventions"): HMF_t
//      stores dn/dlnM on the axis ln(m_h * (Omega_m - Omega_nu)), so the
//      SAME dndlnmh table read under a different (Omega_m - Omega_nu)
//      returns a DIFFERENT value at the same lnM. A backend that had
//      silently un-shifted the axis would make these two agree.
//   4. Per-bin S_stack indexing: the two bins differ by exactly the ratio
//      of their (constant) S_stack planes.
//   5. SelGlWeights's own moments: norm/lnm_eff/mu2/z_eff against closed
//      forms on the same synthetic block.
//   6. The shared Gauss-Legendre primitive y3_pipelines::gl_nodes: exact
//      on polynomials up to degree 2N-1, weights summing to (b - a),
//      nodes symmetric and strictly interior -- the numerical foundation
//      every 0d backend in src/pipelines rests on.
//   7. Guard clauses: bin_index outside the S_stack range throws.
//
// Requires Y3_CLUSTER_CPP_DIR only indirectly (no NFW tables are read
// here -- unlike the shear backends, counts touch no data/ file).
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "models/dv_do_dz_t.hh"
#include "models/hmf_t.hh"
#include "models/omega_z_des.hh"
#include "models/sel_function_t.hh"

#include "pipelines/des_y3/number_counts/cpp/0d/num_counts_sij_gl_t.hh"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

  // A window narrow enough that every GL node stays strictly interior to
  // every synthetic table axis (distances/z is queried WITHOUT clamping by
  // DV_DO_DZ_t, so leaving the range throws rather than misbehaving), and
  // below OMEGA_Z_DES's fit/fit2 seam at z = 0.504.
  double constexpr ZT_LO = 0.10;
  double constexpr ZT_HI = 0.50;
  double constexpr LNM_LO = 30.0;
  double constexpr LNM_HI = 36.0;
  int constexpr N_Z = 6;
  int constexpr N_LNM = 8;

  // The two bins' (constant) S_stack planes.  Distinct so that bin
  // indexing into the selection tensor is genuinely exercised.
  double constexpr S_BIN0 = 2.0;
  double constexpr S_BIN1 = 3.5;

  cosmosis::DataBlock
  make_cfg_block()
  {
    cosmosis::DataBlock cfg;
    char const* label = NumCountsSijGl::module_label();
    cfg.put_val(label, "n_lnm", N_LNM);
    cfg.put_val(label, "n_z", N_Z);
    cfg.put_val(label, "zt_low", ZT_LO);
    cfg.put_val(label, "zt_high", ZT_HI);
    cfg.put_val(label, "lnm_low", LNM_LO);
    cfg.put_val(label, "lnm_high", LNM_HI);
    return cfg;
  }

  // Per-sample block.  dndlnmh is LINEAR in the (already log-shifted) mass
  // axis and constant in z, so HMF_t's interpolation is exact everywhere in
  // range and the mass-shift contract in TEST_CASE 3 is analytically
  // checkable; d_a and sci_average are flat; hmf_s = 0 / hmf_q = 1 (the
  // real fiducial point's cluster_abundance values) collapses HMF_t to the
  // raw table value.
  cosmosis::DataBlock
  make_sample_block(double sci_value = 3.0e-4, double omega_nu = 0.0)
  {
    cosmosis::DataBlock s;

    s.put_val("cosmological_parameters", "omega_m", 0.3);
    s.put_val("cosmological_parameters", "omega_lambda", 0.7);
    s.put_val("cosmological_parameters", "omega_k", 0.0);
    s.put_val("cosmological_parameters", "omega_nu", omega_nu);
    s.put_val("cosmological_parameters", "h0", 0.7);

    s.put_val("cluster_abundance", "hmf_s", 0.0);
    s.put_val("cluster_abundance", "hmf_q", 1.0);
    s.put_val("mass_function", "m_h", std::vector<double>{1.0e13, 1.0e15});
    s.put_val("mass_function", "z", std::vector<double>{0.0, 1.0});
    // (n_z, n_m) row-major; linear in the mass index, flat in z.
    s.put_val("mass_function", "dndlnmh",
              cosmosis::ndarray<double>(
                std::vector<double>{7.0, 2.0, 7.0, 2.0}, {2, 2}));

    s.put_val("distances", "z", std::vector<double>{0.0, 1.0});
    s.put_val("distances", "d_a", std::vector<double>{1500.0, 1500.0});

    s.put_val("average_sigma_crit_inv", "zlense",
              std::vector<double>{0.0, 1.0});
    s.put_val("average_sigma_crit_inv", "sci_average",
              std::vector<double>{sci_value, sci_value});

    s.put_val("sel_function", "lnM", std::vector<double>{LNM_LO, LNM_HI});
    s.put_val("sel_function", "z", std::vector<double>{ZT_LO, ZT_HI});
    std::vector<double> s_stack;
    for (double c : {S_BIN0, S_BIN1})
      for (int i = 0; i != 4; ++i) s_stack.push_back(c);
    s.put_val("sel_function", "S_stack",
              cosmosis::ndarray<double>(s_stack, {2, 2, 2})); // (bin, z, lnm)

    return s;
  }

  // The REQUIRED DataBlock contract of this backend, as an explicit list
  // (see the "required inputs" TEST_CASE below).  `make_sample_block_omitting`
  // rebuilds the sample with exactly one required key left out, so each
  // entry can be checked to fail loudly rather than default silently.
  struct DbKey { char const* section; char const* key; };

  std::vector<DbKey> const REQUIRED_SAMPLE_KEYS{
    {"cosmological_parameters", "omega_m"},
    {"cosmological_parameters", "omega_nu"},
    {"cosmological_parameters", "h0"},
    {"cluster_abundance", "hmf_s"},
    {"cluster_abundance", "hmf_q"},
    {"mass_function", "m_h"},
    {"mass_function", "z"},
    {"mass_function", "dndlnmh"},
    {"distances", "z"},
    {"distances", "d_a"},
    {"sel_function", "lnM"},
    {"sel_function", "z"},
    {"sel_function", "S_stack"},
  };

  bool
  omitted(DbKey const& skip, char const* section, char const* key)
  {
    return std::string(skip.section) == section &&
           std::string(skip.key) == key;
  }

  // Same block as make_sample_block, minus exactly one required key.
  // average_sigma_crit_inv is deliberately absent throughout: counts call
  // build_weights(include_sci = false), so the section must not be needed.
  cosmosis::DataBlock
  make_sample_block_omitting(DbKey const& skip)
  {
    cosmosis::DataBlock s;
    auto keep = [&](char const* sec, char const* key) {
      return !omitted(skip, sec, key);
    };

    if (keep("cosmological_parameters", "omega_m"))
      s.put_val("cosmological_parameters", "omega_m", 0.3);
    s.put_val("cosmological_parameters", "omega_lambda", 0.7);
    s.put_val("cosmological_parameters", "omega_k", 0.0);
    if (keep("cosmological_parameters", "omega_nu"))
      s.put_val("cosmological_parameters", "omega_nu", 0.0);
    if (keep("cosmological_parameters", "h0"))
      s.put_val("cosmological_parameters", "h0", 0.7);

    if (keep("cluster_abundance", "hmf_s"))
      s.put_val("cluster_abundance", "hmf_s", 0.0);
    if (keep("cluster_abundance", "hmf_q"))
      s.put_val("cluster_abundance", "hmf_q", 1.0);
    if (keep("mass_function", "m_h"))
      s.put_val("mass_function", "m_h",
                std::vector<double>{1.0e13, 1.0e15});
    if (keep("mass_function", "z"))
      s.put_val("mass_function", "z", std::vector<double>{0.0, 1.0});
    if (keep("mass_function", "dndlnmh"))
      s.put_val("mass_function", "dndlnmh",
                cosmosis::ndarray<double>(
                  std::vector<double>{7.0, 2.0, 7.0, 2.0}, {2, 2}));

    if (keep("distances", "z"))
      s.put_val("distances", "z", std::vector<double>{0.0, 1.0});
    if (keep("distances", "d_a"))
      s.put_val("distances", "d_a", std::vector<double>{1500.0, 1500.0});

    if (keep("sel_function", "lnM"))
      s.put_val("sel_function", "lnM", std::vector<double>{LNM_LO, LNM_HI});
    if (keep("sel_function", "z"))
      s.put_val("sel_function", "z", std::vector<double>{ZT_LO, ZT_HI});
    if (keep("sel_function", "S_stack")) {
      std::vector<double> s_stack;
      for (double c : {S_BIN0, S_BIN1})
        for (int i = 0; i != 4; ++i) s_stack.push_back(c);
      s.put_val("sel_function", "S_stack",
                cosmosis::ndarray<double>(s_stack, {2, 2, 2}));
    }
    return s;
  }

  // Independent reassembly of N_ij from the same leaf models, following
  // sel_gl_weights.hh's documented formula -- deliberately written as a
  // plain double loop rather than by calling SelGlWeights again.
  double
  reference_counts(cosmosis::DataBlock& s, int bin)
  {
    y3_cluster::HMF_t const hmf(s);
    y3_cluster::DV_DO_DZ_t const dv(s);
    y3_cluster::OMEGA_Z_DES const omega(s);
    y3_cluster::SelFunction_t const sel(s, bin);

    std::vector<double> lnm_x, lnm_w, z_x, z_w;
    y3_pipelines::gl_nodes(LNM_LO, LNM_HI, N_LNM, lnm_x, lnm_w);
    y3_pipelines::gl_nodes(ZT_LO, ZT_HI, N_Z, z_x, z_w);

    double total = 0.0;
    for (std::size_t k = 0; k != lnm_x.size(); ++k) {
      double inner = 0.0;
      for (std::size_t q = 0; q != z_x.size(); ++q) {
        double const z = z_x[q];
        inner += z_w[q] * dv(z) * omega(z) * hmf(lnm_x[k], z) *
                 sel(lnm_x[k], z);
      }
      total += lnm_w[k] * inner;
    }
    return total;
  }

} // namespace

TEST_CASE("NumCountsSijGl matches an independently assembled W_ij(lnM) sum")
{
  auto cfg = make_cfg_block();
  auto s = make_sample_block();

  NumCountsSijGl mod(cfg);
  mod.set_sample(s);

  for (int b : {0, 1}) {
    auto const out = mod.evaluate({static_cast<double>(b)});
    CHECK(out[0] == Approx(reference_counts(s, b)).epsilon(1e-9));
    CHECK(out[0] > 0.0); // sanity: not silently zero
  }
}

TEST_CASE("NumCountsSijGl does not fold in Sigma_crit^-1")
{
  // build_weights(s, include_sci = false) is the ONE structural difference
  // between the counts and shear uses of the shared SelGlWeights core.
  // Changing average_sigma_crit_inv by 3 orders of magnitude must leave
  // the counts bit-identical; the shear side of the same core (pinned by
  // shear1h_gl.test.cc) would scale with it.
  auto cfg = make_cfg_block();
  auto s_lo = make_sample_block(/*sci_value=*/3.0e-4);
  auto s_hi = make_sample_block(/*sci_value=*/3.0e-1);

  NumCountsSijGl a(cfg);
  a.set_sample(s_lo);
  NumCountsSijGl b(cfg);
  b.set_sample(s_hi);

  for (int bin : {0, 1}) {
    auto const va = a.evaluate({static_cast<double>(bin)});
    auto const vb = b.evaluate({static_cast<double>(bin)});
    CHECK(va[0] == Approx(vb[0]).epsilon(1e-12));
  }
}

TEST_CASE("NumCountsSijGl inherits HMF_t's ln(m (Omega_m - Omega_nu)) mass axis")
{
  // CLAUDE.md, "Unit conventions": HMF_t stores dn/dlnM with x-axis
  // ln(m_h * (Omega_m - Omega_nu)).  The SAME dndlnmh table therefore
  // yields a DIFFERENT dn/dlnM at the same lnM once Omega_nu moves, and a
  // backend that had un-shifted the axis would make these two agree.
  auto cfg = make_cfg_block();
  auto s0 = make_sample_block(3.0e-4, /*omega_nu=*/0.0);
  auto s1 = make_sample_block(3.0e-4, /*omega_nu=*/0.05);

  NumCountsSijGl m0(cfg);
  m0.set_sample(s0);
  NumCountsSijGl m1(cfg);
  m1.set_sample(s1);

  auto const v0 = m0.evaluate({0.0});
  auto const v1 = m1.evaluate({0.0});
  CHECK(v0[0] != Approx(v1[0]).epsilon(1e-6));

  // ...and the shift is the documented one, checked directly on HMF_t:
  // the table is linear in the (shifted) log-mass axis, so an exact hand
  // computation is available.
  y3_cluster::HMF_t const hmf0(s0);
  double const ax_lo = std::log(1.0e13 * 0.3);
  double const ax_hi = std::log(1.0e15 * 0.3);
  double const lnM = 33.0;
  double const t = (lnM - ax_lo) / (ax_hi - ax_lo);
  CHECK(hmf0(lnM, 0.5) == Approx(7.0 + t * (2.0 - 7.0)).epsilon(1e-9));

  y3_cluster::HMF_t const hmf1(s1);
  double const ax_lo1 = std::log(1.0e13 * (0.3 - 0.05));
  double const ax_hi1 = std::log(1.0e15 * (0.3 - 0.05));
  double const t1 = (lnM - ax_lo1) / (ax_hi1 - ax_lo1);
  CHECK(hmf1(lnM, 0.5) == Approx(7.0 + t1 * (2.0 - 7.0)).epsilon(1e-9));
}

TEST_CASE("NumCountsSijGl indexes the per-bin S_stack plane")
{
  // Both bins share every other factor and differ only by a constant
  // selection plane, so their counts must be in exactly that ratio.  A
  // backend that ignored bin_index, or that read the tensor with the wrong
  // stride, would not reproduce it.
  auto cfg = make_cfg_block();
  auto s = make_sample_block();

  NumCountsSijGl mod(cfg);
  mod.set_sample(s);

  auto const v0 = mod.evaluate({0.0});
  auto const v1 = mod.evaluate({1.0});
  CHECK(v1[0] / v0[0] == Approx(S_BIN1 / S_BIN0).epsilon(1e-9));
}

TEST_CASE("SelGlWeights moments: norm, lnm_eff, mu2 and z_eff on the synthetic block")
{
  // The moments are what the downstream moment-expansion backends consume,
  // and only `norm` is reachable through NumCountsSijGl::evaluate, so they
  // are pinned here directly against their defining sums.
  auto cfg = make_cfg_block();
  auto s = make_sample_block();

  y3_pipelines::SelGlWeights core(cfg, NumCountsSijGl::module_label());
  core.build_weights(s, /*include_sci=*/false);

  REQUIRE(core.n_bins() == 2);
  REQUIRE(core.lnm_x().size() == static_cast<std::size_t>(N_LNM));
  REQUIRE(core.z_x().size() == static_cast<std::size_t>(N_Z));

  for (int b : {0, 1}) {
    auto const& W = core.weights(b);
    auto const& xw = core.lnm_w();
    auto const& xx = core.lnm_x();

    double n0 = 0.0, n1 = 0.0;
    for (std::size_t k = 0; k != W.size(); ++k) {
      n0 += xw[k] * W[k];
      n1 += xw[k] * W[k] * xx[k];
    }
    CHECK(core.norm(b) == Approx(n0).epsilon(1e-12));
    CHECK(core.lnm_eff(b) == Approx(n1 / n0).epsilon(1e-12));

    double m2 = 0.0;
    for (std::size_t k = 0; k != W.size(); ++k) {
      double const d = xx[k] - core.lnm_eff(b);
      m2 += xw[k] * W[k] * d * d;
    }
    CHECK(core.mu2(b) == Approx(m2 / n0).epsilon(1e-12));

    // Every moment must land inside the integration window; a sign or
    // stride error typically pushes lnm_eff outside it.
    CHECK(core.lnm_eff(b) > LNM_LO);
    CHECK(core.lnm_eff(b) < LNM_HI);
    CHECK(core.z_eff(b) > ZT_LO);
    CHECK(core.z_eff(b) < ZT_HI);
    CHECK(core.mu2(b) > 0.0);
  }

  // norm() IS what evaluate() returns -- the wrapper adds nothing else.
  NumCountsSijGl mod(cfg);
  mod.set_sample(s);
  for (int b : {0, 1})
    CHECK(mod.evaluate({static_cast<double>(b)})[0] ==
          Approx(core.norm(b)).epsilon(1e-12));
}

TEST_CASE("y3_pipelines::gl_nodes is an exact Gauss-Legendre rule")
{
  // The numerical foundation of every 0d backend in src/pipelines: an
  // N-node Gauss-Legendre rule integrates polynomials up to degree 2N-1
  // exactly.  Checked on [a, b] = [-0.7, 1.4], away from the symmetric
  // [-1, 1] case where several classes of bug cancel.
  double const a = -0.7, b = 1.4;
  for (std::size_t N : {2u, 3u, 5u, 8u}) {
    std::vector<double> x, w;
    y3_pipelines::gl_nodes(a, b, N, x, w);
    REQUIRE(x.size() == N);
    REQUIRE(w.size() == N);

    // Weights sum to the interval length.
    double wsum = 0.0;
    for (double wi : w) wsum += wi;
    CHECK(wsum == Approx(b - a).epsilon(1e-13));

    // Nodes strictly interior and strictly increasing.
    for (std::size_t i = 0; i != N; ++i) {
      CHECK(x[i] > a);
      CHECK(x[i] < b);
      if (i) CHECK(x[i] > x[i - 1]);
    }

    // Node set symmetric about the midpoint.
    for (std::size_t i = 0; i != N; ++i)
      CHECK(x[i] + x[N - 1 - i] == Approx(a + b).margin(1e-12));

    // Exact on every monomial up to degree 2N - 1.
    for (std::size_t p = 0; p <= 2 * N - 1; ++p) {
      double quad = 0.0;
      for (std::size_t i = 0; i != N; ++i)
        quad += w[i] * std::pow(x[i], static_cast<double>(p));
      double const exact = (std::pow(b, static_cast<double>(p + 1)) -
                            std::pow(a, static_cast<double>(p + 1))) /
                           static_cast<double>(p + 1);
      CHECK(quad == Approx(exact).margin(1e-11));
    }
  }
}

TEST_CASE("NumCountsSijGl ini-option contract: [NumCountsSijGl] required vs optional")
{
  // The ini side of the contract, cross-checked against the shipped
  // pipelines (cosmosis-models/des_y3.ini, physics_validation_*.ini):
  //
  //   [NumCountsSijGl]
  //   file      = .../des_y3_numcounts_0d_cpp/NumCountsSijGl.so
  //   bin_index = 0 1 ... 11     REQUIRED (the wall; make_grid_points)
  //   zt_low    = 0.05           REQUIRED
  //   zt_high   = 0.80           REQUIRED
  //   lnm_low   = 29.9336        REQUIRED
  //   lnm_high  = 36.7300        REQUIRED
  //   n_lnm     = 96             optional, default 96
  //   n_z       = 64             optional, default 64
  //
  // Each of the four required scalars must fail loudly when absent -- a
  // silent default here would change the integration window (and hence
  // every published count) without any pipeline-visible signal.
  char const* label = NumCountsSijGl::module_label();
  for (char const* missing : {"zt_low", "zt_high", "lnm_low", "lnm_high"}) {
    cosmosis::DataBlock cfg;
    if (std::string(missing) != "zt_low") cfg.put_val(label, "zt_low", ZT_LO);
    if (std::string(missing) != "zt_high") cfg.put_val(label, "zt_high", ZT_HI);
    if (std::string(missing) != "lnm_low") cfg.put_val(label, "lnm_low", LNM_LO);
    if (std::string(missing) != "lnm_high")
      cfg.put_val(label, "lnm_high", LNM_HI);
    cfg.put_val(label, "n_lnm", N_LNM);
    cfg.put_val(label, "n_z", N_Z);
    CHECK_THROWS(NumCountsSijGl{cfg});
  }

  // n_lnm / n_z ARE optional, and their documented defaults are 96 / 64.
  cosmosis::DataBlock cfg_default;
  cfg_default.put_val(label, "zt_low", ZT_LO);
  cfg_default.put_val(label, "zt_high", ZT_HI);
  cfg_default.put_val(label, "lnm_low", LNM_LO);
  cfg_default.put_val(label, "lnm_high", LNM_HI);
  y3_pipelines::SelGlWeights core(cfg_default, label);
  CHECK(core.lnm_x().size() == 96u);
  CHECK(core.z_x().size() == 64u);

  // bin_index is read by make_grid_points, not by the constructor, and it
  // is the wall the ini must supply: no bin_index, no grid.
  cosmosis::DataBlock cfg_no_wall;
  cfg_no_wall.put_val(label, "zt_low", ZT_LO);
  cfg_no_wall.put_val(label, "zt_high", ZT_HI);
  cfg_no_wall.put_val(label, "lnm_low", LNM_LO);
  cfg_no_wall.put_val(label, "lnm_high", LNM_HI);
  CHECK_THROWS(NumCountsSijGl::make_grid_points(cfg_no_wall));

  cosmosis::DataBlock cfg_wall = cfg_no_wall;
  cfg_wall.put_val(label, "bin_index", std::vector<double>{0.0, 1.0, 2.0});
  auto const grid = NumCountsSijGl::make_grid_points(cfg_wall);
  REQUIRE(grid.size() == 3);
  CHECK(grid.points[0][0] == Approx(0.0));
  CHECK(grid.points[2][0] == Approx(2.0));
  CHECK(grid.names == std::vector<std::string>{"bin_index"});
}

TEST_CASE("NumCountsSijGl DataBlock contract: every required sample key fails loudly")
{
  // The sample side of the contract.  Counts need exactly these sections:
  //   cosmological_parameters (omega_m, omega_nu, h0)  -- HMF axis + DV
  //   cluster_abundance       (hmf_s, hmf_q)           -- HMF_t amplitude
  //   mass_function           (m_h, z, dndlnmh)        -- dn/dlnM
  //   distances               (z, d_a)                 -- dV/dOmegadz
  //   sel_function            (lnM, z, S_stack)        -- S_ij
  // and, crucially, NOT average_sigma_crit_inv.  Dropping any one of the
  // above must throw rather than silently produce a number.
  auto cfg = make_cfg_block();

  for (auto const& k : REQUIRED_SAMPLE_KEYS) {
    auto s = make_sample_block_omitting(k);
    NumCountsSijGl mod(cfg);
    INFO("omitted " << k.section << "/" << k.key);
    CHECK_THROWS(mod.set_sample(s));
  }

  // The complete block (still with NO average_sigma_crit_inv section at
  // all) must succeed -- this is what makes the counts/shear split a
  // contract rather than a comment.
  auto s_full =
    make_sample_block_omitting(DbKey{"nothing_at_all", "nothing_at_all"});
  REQUIRE_FALSE(s_full.has_val("average_sigma_crit_inv", "zlense"));
  NumCountsSijGl mod(cfg);
  REQUIRE_NOTHROW(mod.set_sample(s_full));
  CHECK(mod.evaluate({0.0})[0] > 0.0);
}

TEST_CASE("NumCountsSijGl guard clause: bin_index outside the S_stack range")
{
  auto cfg = make_cfg_block();
  auto s = make_sample_block();
  NumCountsSijGl mod(cfg);
  mod.set_sample(s);

  CHECK_THROWS_AS(mod.evaluate({2.0}), std::out_of_range);
  CHECK_THROWS_AS(mod.evaluate({-1.0}), std::out_of_range);
}

TEST_CASE("NumCountsSijGl publishes its own module label and output section")
{
  // Backends of one stage share an output section and CosmoSIS put_val
  // does NOT overwrite (CLAUDE.md): this backend must keep its own
  // section so it can co-run with production NumCountsSel.so.
  CHECK(std::string(NumCountsSijGl::module_label()) == "NumCountsSijGl");
  REQUIRE(NumCountsSijGl::n_outputs == 1);
  CHECK(std::string(NumCountsSijGl::output_sections()[0]) ==
        "numcounts_sij_gl");
}
