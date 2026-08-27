#include "catch2/catch.hpp"
#include "models/hmf_t.hh"
#include "utils/make_ifstream.hh"
#include "utils/read_vector.hh"

// This is the code we're actually testing
#include "models/nfw_dsigma_mis.cuh"
#include "utils/cuda_interp_2d.cuh"

// Host twin, for the r_s() device-vs-host parity check below.
#include "models/nfw_dsigma_mis.hh"
#include "utils/interp_1d.hh"

#include <fstream>
#include <iomanip>
#include <fmt/format.h>

using y3_cluster::HMF_t;
// using y3_cluster::Interp2D;
using y3_cuda::NFW_DSIGMA_MIS;

TEST_CASE("Test NFW Misc Implementation")
{
  // Setup the initial conditions properly
  double  conc = 4.0;
  double  omega_m = 0.3;
  double  RHOC = 2.77533742639e+11;
  double  RHOM = RHOC*omega_m;
  NFW_DSIGMA_MIS model(conc, RHOM, "gamma");

  // FINDING (real, at the project's standard 1e-3 -- see
  // docs/known_issues/nfw_dsigma_mis_defect.md): authored at epsrel=5e-3 because the
  // (r=10, r_mis=2.0) point below does not pass at 1e-3 (measured
  // -0.497% vs the cluster_toolkit reference; not a table-domain-edge
  // artifact -- x=r/rs~35.5, x_mis=r_mis/rs~7.1 are both well inside
  // the tabulated x in [1e-3, 5e3] range). Kept strict rather than
  // loosened, per policy: a failing test here is doing its job.
  double const epsrel = 1.0e-3;
  double const epsabs = 1.0e-12;

  // Here we need to fill the vectors rs, rmiss, lnMs and answers 
  // Radii vector log spaced from 0.1 to 10 with 10 points
  std::vector<double> r{ 0.1       ,  0.16681005,  0.27825594,  0.46415888,  0.77426368,
                         1.29154967,  2.15443469,  3.59381366,  5.9948425 , 10.        };

  // Answers drawn from cluster toolkit
  // Each columns is a different rmis, each row is a different r
  std::vector<double> answers{12.3685326, 0.3899747, 0.0104446,
                              22.7491859, 0.9300333, 0.0264995,
                              33.3082693, 2.0334094, 0.0641354,
                              36.0917619, 3.9092169, 0.1455426,
                              28.3882168, 6.2393766, 0.3031914,
                              17.5178459, 7.6804544, 0.5638920,
                              9.4010319, 6.7788772, 0.9002713,
                              4.6358405, 4.2448975, 1.1650680,
                              2.1556775, 2.1118245, 1.1349319,
                              0.9599990, 0.9545601, 0.7860161};

  std::vector<double> rmis{0.1,0.5,2.0};
  std::vector<double> lnMs{32.2361913};

  std::size_t const n_r = r.size();
  std::size_t const n_rmiss = rmis.size();
  std::size_t const n_lnMs = lnMs.size();

  auto out = y3_cluster::make_ofstream("data/nfw_dsigma_mis_test.out");
  REQUIRE(out.good());
  out << std::setw(16);
  out << std::setprecision(16);
  out << "r\trmis\tlnM\tytrue\tytest\n";
  
  for (std::size_t i = 0; i != n_r; ++i) {
    for (std::size_t j = 0; j != n_rmiss; ++j) {
      for (std::size_t k = 0; k != n_lnMs; ++k) {
        double const calculated_result = model(r[i], rmis[j], lnMs[k]);
        // Adjust this look to look up the answers correctly; the indexing needed will depend
        // on the order of storage of the results.
        double const expected_result = answers[k + n_lnMs*j + n_lnMs*n_rmiss*i];
        CHECK(calculated_result == Approx(expected_result).epsilon(epsrel).margin(epsabs));
        out << r[i] << '\t'
            << rmis[j] << '\t'
            << lnMs[k] << '\t'
            << expected_result << '\t'
            << calculated_result << '\n';
      }
    }
  }
}

// y3_cuda::NFW_DSIGMA_MIS::r_s(lnM) was added 2026-08-27 (issue #24
// debugging: shear_prj_frozen_gpu_t.cuh's cl-channel amplitude anchor
// was hand-computing r_200/(fixed c=4) with a bare rho_crit constant
// instead of calling this method, diverging from the profile's actual
// rho_ref/per-mass concentration -- see
// docs/known_issues/frozen_physics_signed_rnd_defect.md). Pin it
// against the CPU twin (nfw_dsigma_mis.hh:129) so the two can't drift
// apart again, with and without a concentration table set.
TEST_CASE("y3_cuda::NFW_DSIGMA_MIS::r_s matches the host twin exactly")
{
  double const conc = 4.0;
  double const rho_ref = 2.77533742639e+11 * 0.3;  // omega_m * rho_crit,0

  SECTION("fixed concentration (no table set)")
  {
    y3_cluster::NFW_DSIGMA_MIS host(conc, rho_ref, "single");
    NFW_DSIGMA_MIS dev(conc, rho_ref, y3_cuda::DSIGMA_MIS_SINGLE);

    for (double log10_M : {12.0, 13.0, 14.0, 14.5, 15.0, 15.5}) {
      double const lnM = std::log(std::pow(10.0, log10_M));
      CHECK(dev.r_s(lnM) == Approx(host.r_s(lnM)).epsilon(1.0e-12));
    }
  }

  SECTION("per-mass concentration table set")
  {
    std::vector<double> const lnM_tab{28.0, 30.0, 32.0, 34.0, 36.0};
    std::vector<double> const conc_tab{6.0, 5.5, 5.0, 4.2, 3.5};

    y3_cluster::NFW_DSIGMA_MIS host(conc, rho_ref, "single");
    host.set_concentration_table(y3_cluster::Interp1D(lnM_tab, conc_tab));

    NFW_DSIGMA_MIS dev(conc, rho_ref, y3_cuda::DSIGMA_MIS_SINGLE);
    dev.set_concentration_table(lnM_tab, conc_tab);

    for (double lnM : {29.0, 31.0, 33.0, 35.0}) {
      CHECK(dev.conc_at(lnM) == Approx(host.conc_at(lnM)).epsilon(1.0e-12));
      CHECK(dev.r_s(lnM) == Approx(host.r_s(lnM)).epsilon(1.0e-12));
    }
  }
}

// 
// To Generate The Answers; Go To a Python Interface with Cluster Toolkit
// 
// import numpy as np
// import cluster_toolkit as ct

// def make_deltasimga_mis(R, Rmis=0.3, Mt=1e14, c=4., omega_m=0.3):
//     Rp = np.logspace(np.log10(0.001),np.log10(15.),300)
//     st = ct.deltasigma.Sigma_nfw_at_R(Rp, Mt, c, omega_m)
//     st_mis = ct.miscentering.Sigma_mis_at_R(Rp, Rp, st, Mt, c, omega_m, Rmis, kernel='gamma')
//     _dst_mis = ct.miscentering.DeltaSigma_mis_at_R(Rp, Rp, st_mis)
//     return np.interp(R,Rp,_dst_mis)
    
// R = np.logspace(np.log10(0.1),np.log10(10.),10)
// R

// res = np.vstack([make_deltasimga_mis(R, 0.1),
//                  make_deltasimga_mis(R, 0.5),
//                  make_deltasimga_mis(R, 2.)])

// for row in res.T:
//     print('%.7f, %.7f, %.7f,'%(row[0],row[1],row[2]))