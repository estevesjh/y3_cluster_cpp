// GPU module smoke test for ShearPrjFrozenGpu.so (des_y3 Phase 2 rollout).
//
// Unlike the header-only device models exercised elsewhere in this suite
// (nfw_dsigma_mis.test.cu etc.), ShearPrjFrozenGpu.cu has no adjacent
// header -- its class is defined directly in the .cu translation unit and
// only exposed via the CosmoSIS C-ABI (setup/execute/cleanup) that
// DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE emits. So this test drives the
// actual built .so through that C-ABI with a real cosmosis::DataBlock,
// built from docs/figs/real_pipeline_extract_prj2h_output -- regenerable
// via `cosmosis docs/figs/real_pipeline_extract_prj2h.ini` (see
// Y3_CLUSTER_CPP_DIR/CLAUDE.md), unlike a personal scratch dump. That
// same dump's [shear_prj_frozen_physics] section ran with this test's
// exact module config (n_lnm 16, same 180-point wall), through
// ShearPrjFrozenPhysics.so -- the CPU counterpart of this GPU port -- so
// dsigma_prj_frozen_physics/{vals,rnd,cl} is a same-algorithm CPU
// reference, not a looser cross-approximation bound (that comparison,
// vs the n_lnm=24 exact dsigma_prj evaluator, is what the module's
// README quotes separately).
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/datablock_status.h"
#include "cosmosis/datablock/ndarray.hh"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

  std::string
  dump_dir()
  {
    char const* repo = std::getenv("Y3_CLUSTER_CPP_DIR");
    REQUIRE(repo != nullptr);
    return std::string(repo) + "/docs/figs/real_pipeline_extract_prj2h_output";
  }

  // One value per (non-comment) line.
  std::vector<double>
  read_column(std::string const& path)
  {
    std::ifstream in(path);
    REQUIRE(in.good());
    std::vector<double> out;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      out.push_back(std::stod(line));
    }
    return out;
  }

  // A comment line followed by N rows of whitespace-separated values,
  // flattened row-major -- the layout the test-sampler writer uses for a
  // 2D ndarray (shape (n_rows, n_cols), y-axis slow, x-axis fast; matches
  // the make_Interp2D/Interp2D(xs, ys, ndarray) convention used throughout
  // this repo).
  std::vector<double>
  read_matrix_flat(std::string const& path, std::size_t& n_rows)
  {
    std::ifstream in(path);
    REQUIRE(in.good());
    std::vector<double> flat;
    n_rows = 0;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      double v;
      while (ss >> v) flat.push_back(v);
      ++n_rows;
    }
    return flat;
  }

  double
  read_scalar(std::string const& values_path, std::string const& key)
  {
    std::ifstream in(values_path);
    REQUIRE(in.good());
    std::string line;
    while (std::getline(in, line)) {
      auto const eq = line.find('=');
      if (eq == std::string::npos) continue;
      std::string k = line.substr(0, eq);
      while (!k.empty() && std::isspace(static_cast<unsigned char>(k.back())))
        k.pop_back();
      if (k == key) return std::stod(line.substr(eq + 1));
    }
    FAIL("key '" + key + "' not found in " + values_path);
    return 0.0;
  }

  cosmosis::ndarray<double>
  read_2d(std::string const& dir, std::string const& file, std::size_t n_x)
  {
    std::size_t n_y = 0;
    auto flat = read_matrix_flat(dir + "/" + file, n_y);
    REQUIRE(flat.size() == n_y * n_x);
    return cosmosis::ndarray<double>(flat, {n_y, n_x});
  }

} // namespace

TEST_CASE("ShearPrjFrozenGpu.so matches its CPU counterpart "
         "ShearPrjFrozenPhysics.so")
{
  std::string const D = dump_dir();
  {
    std::ifstream probe(D + "/mass_function/m_h.txt");
    if (!probe.good()) {
      WARN("skipping: reference dump not present at " << D
           << " -- run `cosmosis docs/figs/real_pipeline_extract_prj2h.ini`");
      return;
    }
  }

  // ---- cfg DataBlock: module knobs + the production 180-point wall,
  // mirroring [shear_prj_frozen_physics] in
  // real_pipeline_extract_prj2h.ini exactly (same n_lnm, same wall --
  // that section is this GPU port's CPU counterpart on this sample). ----
  cosmosis::DataBlock cfg;
  char const* mod = "ShearPrjFrozenGpu";
  cfg.put_val(mod, "zt_low", 0.10);
  cfg.put_val(mod, "zt_high", 0.75);
  cfg.put_val(mod, "lnm_low", 29.9336);
  cfg.put_val(mod, "lnm_high", 35.6814);
  cfg.put_val(mod, "R_max_cMpch", 35.0);
  cfg.put_val(mod, "n_lnm", 16);
  cfg.put_val(mod, "n_per_seg", 10);
  cfg.put_val(mod, "n_zring", 20);
  cfg.put_val(mod, "n_zouter", 20);
  cfg.put_val(mod, "include_omega_z", 0);

  std::vector<double> const RADII{0.0426,  0.0669,  0.1045,  0.1652, 0.2607,
                                  0.4117,  0.6505,  1.0257,  1.6181, 2.5537,
                                  4.0265,  6.3490,  10.0107, 15.7832, 24.8771};
  std::vector<double> const ZLO{0.20, 0.35, 0.50}, ZHI{0.35, 0.50, 0.65};
  std::vector<double> lambda_bin, zo_low, zo_high, radii;
  for (int zb = 0; zb != 3; ++zb)
    for (int lb = 0; lb != 4; ++lb)
      for (double r : RADII) {
        lambda_bin.push_back(double(lb));
        zo_low.push_back(ZLO[zb]);
        zo_high.push_back(ZHI[zb]);
        radii.push_back(r);
      }
  REQUIRE(radii.size() == 180u);
  cfg.put_val(mod, "lambda_bin", lambda_bin);
  cfg.put_val(mod, "zo_low", zo_low);
  cfg.put_val(mod, "zo_high", zo_high);
  cfg.put_val(mod, "radii", radii);

  // ---- sample DataBlock: everything set_sample() reads, from the dump. ----
  cosmosis::DataBlock sample;

  auto const m_h = read_column(D + "/mass_function/m_h.txt");
  auto const mf_z = read_column(D + "/mass_function/z.txt");
  sample.put_val("mass_function", "m_h", m_h);
  sample.put_val("mass_function", "z", mf_z);
  sample.put_val("mass_function", "dndlnmh",
                 read_2d(D, "mass_function/dndlnmh.txt", m_h.size()));
  sample.put_val("cluster_abundance", "hmf_s",
                 read_scalar(D + "/cluster_abundance/values.txt", "hmf_s"));
  sample.put_val("cluster_abundance", "hmf_q",
                 read_scalar(D + "/cluster_abundance/values.txt", "hmf_q"));

  auto const hm_lnm = read_column(D + "/halomodel/lnm.txt");
  auto const hm_z = read_column(D + "/halomodel/z.txt");
  sample.put_val("haloModel", "lnM", hm_lnm);
  sample.put_val("haloModel", "z", hm_z);
  sample.put_val("haloModel", "bias",
                 read_2d(D, "halomodel/bias.txt", hm_lnm.size()));

  auto const xi_r = read_column(D + "/xi_nl/r.txt");
  auto const xi_z = read_column(D + "/xi_nl/z.txt");
  sample.put_val("xi_nl", "r", xi_r);
  sample.put_val("xi_nl", "z", xi_z);
  sample.put_val("xi_nl", "xi_nl", read_2d(D, "xi_nl/xi_nl.txt", xi_r.size()));

  sample.put_val("distances", "z", read_column(D + "/distances/z.txt"));
  sample.put_val("distances", "d_a", read_column(D + "/distances/d_a.txt"));
  sample.put_val("distances", "d_c", read_column(D + "/distances/d_c.txt"));

  sample.put_val("average_sigma_crit_inv", "zlense",
                 read_column(D + "/average_sigma_crit_inv/zlense.txt"));
  sample.put_val("average_sigma_crit_inv", "sci_average",
                 read_column(D + "/average_sigma_crit_inv/sci_average.txt"));

  auto const bs_lob = read_column(D + "/b_sel_marginalised/lob.txt");
  auto const bs_zob = read_column(D + "/b_sel_marginalised/zob.txt");
  sample.put_val("b_sel_marginalised", "lob", bs_lob);
  sample.put_val("b_sel_marginalised", "zob", bs_zob);
  sample.put_val("b_sel_marginalised", "b_small",
                 read_2d(D, "b_sel_marginalised/b_small.txt", bs_lob.size()));
  sample.put_val("b_sel_marginalised", "b_large",
                 read_2d(D, "b_sel_marginalised/b_large.txt", bs_lob.size()));

  std::string const cp = D + "/cosmological_parameters/values.txt";
  sample.put_val("cosmological_parameters", "omega_m", read_scalar(cp, "omega_m"));
  sample.put_val("cosmological_parameters", "omega_M", read_scalar(cp, "omega_m"));
  sample.put_val("cosmological_parameters", "omega_lambda",
                 read_scalar(cp, "omega_lambda"));
  sample.put_val("cosmological_parameters", "omega_k", read_scalar(cp, "omega_k"));
  sample.put_val("cosmological_parameters", "omega_nu", read_scalar(cp, "omega_nu"));
  sample.put_val("cosmological_parameters", "h0", read_scalar(cp, "h0"));

  // ---- dlopen the actual built .so and drive it through the real
  // CosmoSIS C-ABI (setup/execute/cleanup), exactly as CosmoSIS would. ----
  char const* so_path_env = std::getenv("SHEAR_PRJ_FROZEN_GPU_SO");
  std::string const so_path =
    so_path_env ? so_path_env
                : "/pscratch/sd/j/jesteves/github/y3_cluster_cpp/gpu-build/"
                  "src/modules/des_y3_shear_prj_frozen_cuda/"
                  "ShearPrjFrozenGpu.so";
  void* handle = dlopen(so_path.c_str(), RTLD_NOW);
  if (!handle) FAIL("dlopen(" << so_path << ") failed: " << dlerror());
  REQUIRE(handle != nullptr);

  using setup_fn_t = void* (*)(cosmosis::DataBlock*);
  using execute_fn_t = DATABLOCK_STATUS (*)(cosmosis::DataBlock*, void*);
  using cleanup_fn_t = int (*)(void*);
  auto setup_fn = reinterpret_cast<setup_fn_t>(dlsym(handle, "setup"));
  auto execute_fn = reinterpret_cast<execute_fn_t>(dlsym(handle, "execute"));
  auto cleanup_fn = reinterpret_cast<cleanup_fn_t>(dlsym(handle, "cleanup"));
  REQUIRE(setup_fn != nullptr);
  REQUIRE(execute_fn != nullptr);
  REQUIRE(cleanup_fn != nullptr);

  void* module = setup_fn(&cfg);
  REQUIRE(module != nullptr);
  DATABLOCK_STATUS const rc = execute_fn(&sample, module);
  REQUIRE(rc == DBS_SUCCESS);

  // Outputs are written as ndarray<double> (CosmoSISScalarEvaluatorModule
  // wraps each buffer that way), not std::vector<double>.
  auto const& nd_vals =
    sample.view<cosmosis::ndarray<double>>("dsigma_prj_frozen_gpu", "vals");
  auto const& nd_rnd =
    sample.view<cosmosis::ndarray<double>>("dsigma_prj_frozen_gpu", "rnd");
  auto const& nd_cl =
    sample.view<cosmosis::ndarray<double>>("dsigma_prj_frozen_gpu", "cl");
  std::vector<double> const got_vals(nd_vals.begin(), nd_vals.end());
  std::vector<double> const got_rnd(nd_rnd.begin(), nd_rnd.end());
  std::vector<double> const got_cl(nd_cl.begin(), nd_cl.end());
  REQUIRE(got_vals.size() == 180u);

  cleanup_fn(module);
  dlclose(handle);

  // Reference: ShearPrjFrozenPhysics.so (the CPU counterpart of this GPU
  // port) on the identical sample and wall, recorded in the dump by
  // real_pipeline_extract_prj2h.ini's [shear_prj_frozen_physics] section.
  auto const ref_vals = read_column(D + "/dsigma_prj_frozen_physics/vals.txt");
  auto const ref_rnd = read_column(D + "/dsigma_prj_frozen_physics/rnd.txt");
  auto const ref_cl = read_column(D + "/dsigma_prj_frozen_physics/cl.txt");
  REQUIRE(ref_vals.size() == 180u);

  // Spot-check across bins/z/radii rather than all 180 -- one per lambda
  // bin, spanning small/large radius and each z slice.
  std::vector<std::size_t> const rows{0, 14, 15, 59, 89, 104, 149, 179};
  for (auto const r : rows) {
    CHECK(got_vals[r] == Approx(ref_vals[r]).epsilon(1.0e-6));
    CHECK(got_rnd[r] == Approx(ref_rnd[r]).epsilon(1.0e-6).margin(1.0e-12));
    CHECK(got_cl[r] == Approx(ref_cl[r]).epsilon(1.0e-6).margin(1.0e-12));
  }
}
