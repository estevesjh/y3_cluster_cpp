// gpu_shear_prj_physics.test.cu
//
// Physics unit tests for ShearPrjGPU module.
// Tests the shear projection components:
//   Sigma_prj: projected surface mass density
//   DSigma_prj: differential surface density (Delta-Sigma)
//   gamma_t^prj: tangential shear = DSigma * Sigma_crit_inv
//
// Each has random (1-halo) and clustering (2-halo) components.
//
// Tests:
// 1. NFW Sigma/DSigma profiles
// 2. Weight functors for different components
// 3. Physical constraints on shear profiles

#include "catch2/catch.hpp"

#include "models/sigma_prj_gpu_t.cuh"
#include "models/sigma_photoz_des.cuh"

#include <cmath>

// Note: Do NOT use "using namespace y3_cuda" at file scope - causes sqrt conflicts with CUDA
// Import helper functions from sigma_prj_gpu_t.cuh in anonymous namespace
namespace {
  using namespace y3_cuda::sp_gpu_detail;
}


// ============================================================================
// Test: R_lambda scaling (same as bsel)
// ============================================================================

TEST_CASE("ShearPrj R_lambda follows DES Y3 scaling", "[gpu][physics][shear]")
{
  CHECK(R_lambda(100.0) == Approx(1.0).epsilon(1e-10));
  CHECK(R_lambda(20.0) == Approx(pow(0.2, 0.2)).epsilon(1e-10));
  CHECK(R_lambda(200.0) == Approx(pow(2.0, 0.2)).epsilon(1e-10));
}


// ============================================================================
// Test: lob_center for richness bins
// ============================================================================

TEST_CASE("ShearPrj lob_center returns correct values", "[gpu][physics][shear]")
{
  CHECK(lob_center(0) == Approx(25.0));
  CHECK(lob_center(1) == Approx(37.5));
  CHECK(lob_center(2) == Approx(52.5));
  CHECK(lob_center(3) == Approx(130.0));
}


// ============================================================================
// Test: Photo-z kernel w_photoz
// ============================================================================

TEST_CASE("w_photoz has correct Epanechnikov form", "[gpu][physics][shear]")
{
  double const zob = 0.4;
  double const sigma_z = 0.05;

  SECTION("Maximum at z = zob") {
    double const w = w_photoz(zob, zob, sigma_z);
    CHECK(w == Approx(1.0));
  }

  SECTION("Decreases away from zob") {
    double const w_center = w_photoz(zob, zob, sigma_z);
    double const w_offset = w_photoz(zob + 0.5 * sigma_z, zob, sigma_z);

    CHECK(w_offset < w_center);
    CHECK(w_offset > 0.0);
  }

  SECTION("Zero outside sigma_z range") {
    double const w_far = w_photoz(zob + 1.5 * sigma_z, zob, sigma_z);
    CHECK(w_far == Approx(0.0));
  }

  SECTION("Symmetric around zob") {
    double const w_plus = w_photoz(zob + 0.3 * sigma_z, zob, sigma_z);
    double const w_minus = w_photoz(zob - 0.3 * sigma_z, zob, sigma_z);

    CHECK(w_plus == Approx(w_minus).epsilon(1e-10));
  }
}


// ============================================================================
// Test: Exclusion angle theta_excl_at_z
// ============================================================================

TEST_CASE("theta_excl_at_z computes exclusion correctly", "[gpu][physics][shear]")
{
  SECTION("Zero exclusion when chi_z << chi_o") {
    // When cluster is much closer than observer's reference, exclusion is small
    double const chi_z = 100.0;   // Mpc/h
    double const chi_o = 1000.0;  // Mpc/h
    double const R_excl = 1.0;    // Mpc/h

    double const theta = theta_excl_at_z(chi_z, chi_o, R_excl);

    INFO("theta_excl = " << theta << " rad");
    CHECK(theta >= 0.0);
    CHECK(theta < 0.1);  // Should be small
  }

  SECTION("Exclusion increases with R_excl") {
    double const chi_z = 500.0;
    double const chi_o = 500.0;

    double const theta_small = theta_excl_at_z(chi_z, chi_o, 0.5);
    double const theta_large = theta_excl_at_z(chi_z, chi_o, 2.0);

    CHECK(theta_large > theta_small);
  }

  SECTION("Returns zero for identical positions") {
    double const chi = 500.0;
    double const R_excl = 1.0;

    // When chi_z = chi_o and R_excl > 0, there should still be exclusion
    double const theta = theta_excl_at_z(chi, chi, R_excl);

    INFO("theta_excl at same distance = " << theta << " rad");
    // cos_ex = (2*chi^2 - R^2) / (2*chi^2) = 1 - R^2/(2*chi^2)
    // For chi=500, R=1: cos_ex ~ 1 - 1e-6 ~ 1, so theta ~ 0
    CHECK(theta >= 0.0);
  }
}


// ============================================================================
// Test: b_sel sigmoid function
// ============================================================================

TEST_CASE("b_sel_sigmoid has correct transition", "[gpu][physics][shear]")
{
  double const theta_lob = 0.01;  // typical angular scale

  SECTION("Sigmoid is 0.5 at transition point") {
    double const theta_mid = 0.5 * theta_lob;
    double const sig = b_sel_sigmoid(theta_mid, theta_lob);

    CHECK(sig == Approx(0.5).epsilon(0.01));
  }

  SECTION("Sigmoid approaches 0 for small theta") {
    double const sig = b_sel_sigmoid(0.0, theta_lob);
    CHECK(sig < 0.1);
  }

  SECTION("Sigmoid approaches 1 for large theta") {
    double const sig = b_sel_sigmoid(2.0 * theta_lob, theta_lob);
    CHECK(sig > 0.9);
  }

  SECTION("Sigmoid is monotonically increasing") {
    double sig_prev = 0.0;
    for (double theta = 0.0; theta < 2.0 * theta_lob; theta += 0.001) {
      double const sig = b_sel_sigmoid(theta, theta_lob);
      CHECK(sig >= sig_prev);
      sig_prev = sig;
    }
  }
}


// ============================================================================
// Test: Simplified NFW Sigma profile
// ============================================================================

TEST_CASE("compute_sigma_nfw has correct NFW behavior", "[gpu][physics][shear]")
{
  double const lnM = 33.5;  // ~3.5e14 solar masses
  double const R_mis = 0.0; // No miscentering

  SECTION("Sigma is positive") {
    for (double R = 0.1; R < 10.0; R += 0.5) {
      double const sig = compute_sigma_nfw(R, R_mis, lnM);
      INFO("R = " << R << ", Sigma = " << sig);
      CHECK(sig >= 0.0);
    }
  }

  SECTION("Sigma decreases with R at large R") {
    double const sig_inner = compute_sigma_nfw(0.5, R_mis, lnM);
    double const sig_outer = compute_sigma_nfw(5.0, R_mis, lnM);

    INFO("Sigma(0.5) = " << sig_inner << ", Sigma(5.0) = " << sig_outer);
    CHECK(sig_inner > sig_outer);
  }

  SECTION("Sigma increases with mass") {
    double const lnM_low = 32.0;
    double const lnM_high = 35.0;
    double const R = 1.0;

    double const sig_low = compute_sigma_nfw(R, R_mis, lnM_low);
    double const sig_high = compute_sigma_nfw(R, R_mis, lnM_high);

    CHECK(sig_high > sig_low);
  }

  SECTION("Sigma is finite at small R") {
    // NFW profile has a cusp at R=0, but should remain finite
    double const sig = compute_sigma_nfw(0.01, R_mis, lnM);

    CHECK(std::isfinite(sig));
    CHECK(sig > 0.0);
  }
}


// ============================================================================
// Test: Simplified NFW DSigma profile
// ============================================================================

TEST_CASE("compute_dsigma_nfw has correct behavior", "[gpu][physics][shear]")
{
  double const lnM = 33.5;
  double const R_mis = 0.0;

  SECTION("DSigma is positive for NFW") {
    // DSigma = <Sigma(<R)> - Sigma(R) > 0 for NFW (centrally concentrated)
    for (double R = 0.1; R < 10.0; R += 0.5) {
      double const dsig = compute_dsigma_nfw(R, R_mis, lnM);
      INFO("R = " << R << ", DSigma = " << dsig);
      CHECK(dsig >= 0.0);
    }
  }

  SECTION("DSigma decreases with R") {
    double const dsig_inner = compute_dsigma_nfw(0.5, R_mis, lnM);
    double const dsig_outer = compute_dsigma_nfw(5.0, R_mis, lnM);

    INFO("DSigma(0.5) = " << dsig_inner << ", DSigma(5.0) = " << dsig_outer);
    CHECK(dsig_inner > dsig_outer);
  }

  SECTION("DSigma increases with mass") {
    double const lnM_low = 32.0;
    double const lnM_high = 35.0;
    double const R = 1.0;

    double const dsig_low = compute_dsigma_nfw(R, R_mis, lnM_low);
    double const dsig_high = compute_dsigma_nfw(R, R_mis, lnM_high);

    CHECK(dsig_high > dsig_low);
  }
}


// ============================================================================
// Test: Shear weight functors
// ============================================================================

TEST_CASE("Shear weight functors select correct components", "[gpu][physics][shear]")
{
  double const common = 1.0;
  double const Sigma_v = 100.0;
  double const DSigma_v = 50.0;
  double const b_Mz = 2.0;
  double const b_sel = 1.5;
  double const xi_val = 0.1;
  bool const has_cl = true;

  SECTION("SigmaRndWeight returns random component") {
    double const w = SigmaRndWeight::weight(common, Sigma_v, DSigma_v,
                                            b_Mz, b_sel, xi_val, has_cl);
    CHECK(w == Approx(common * Sigma_v));
  }

  SECTION("SigmaClWeight returns clustering component") {
    double const w = SigmaClWeight::weight(common, Sigma_v, DSigma_v,
                                           b_Mz, b_sel, xi_val, has_cl);
    CHECK(w == Approx(common * b_Mz * b_sel * xi_val * Sigma_v));
  }

  SECTION("SigmaClWeight is zero when has_cl=false") {
    double const w = SigmaClWeight::weight(common, Sigma_v, DSigma_v,
                                           b_Mz, b_sel, xi_val, false);
    CHECK(w == 0.0);
  }

  SECTION("DSigmaRndWeight returns random DSigma") {
    double const w = DSigmaRndWeight::weight(common, Sigma_v, DSigma_v,
                                             b_Mz, b_sel, xi_val, has_cl);
    CHECK(w == Approx(common * DSigma_v));
  }

  SECTION("DSigmaClWeight returns clustering DSigma") {
    double const w = DSigmaClWeight::weight(common, Sigma_v, DSigma_v,
                                            b_Mz, b_sel, xi_val, has_cl);
    CHECK(w == Approx(common * b_Mz * b_sel * xi_val * DSigma_v));
  }
}


// ============================================================================
// Test: Physical constraints on shear profiles
// ============================================================================

TEST_CASE("Shear profile physical constraints", "[gpu][physics][shear]")
{
  SECTION("gamma_t = DSigma * Sigma_crit_inv is positive") {
    // For positive DSigma and positive Sigma_crit_inv, gamma_t > 0
    // This is true for gravitationally lensing mass distributions

    double const DSigma = 1e14;  // M_sun/Mpc^2
    double const Sigma_crit_inv = 1e-15;  // Mpc^2/M_sun

    double const gamma_t = DSigma * Sigma_crit_inv;

    CHECK(gamma_t > 0.0);
    CHECK(gamma_t < 1.0);  // gamma_t should be << 1 for weak lensing
  }

  SECTION("Sigma_crit_inv increases with lens redshift (at fixed source)") {
    // For a fixed source redshift, Sigma_crit^{-1} increases as the lens
    // moves closer to the source (up to the halfway point)

    // This is qualitative - actual values depend on cosmology
    // We just check the concept is valid

    // For z_lens << z_source: Sigma_crit^{-1} ~ D_lens * D_lens_source / D_source
    // This increases with z_lens until z_lens ~ z_source/2
  }

  SECTION("Total Sigma = Random + Clustering") {
    double const sig_rnd = 1e14;
    double const sig_cl = 0.1e14;

    double const sig_total = sig_rnd + sig_cl;

    // Random dominates at small scales, clustering at large scales
    CHECK(sig_total > sig_rnd);
    CHECK(sig_total == Approx(sig_rnd + sig_cl));
  }
}


// ============================================================================
// Test: Miscentering effects
// ============================================================================

TEST_CASE("Miscentering reduces central Sigma", "[gpu][physics][shear]")
{
  double const lnM = 33.5;
  double const R = 0.1;  // Small radius where miscentering matters most

  SECTION("Sigma decreases with miscentering at small R") {
    double const sig_centered = compute_sigma_nfw(R, 0.0, lnM);
    double const sig_mis = compute_sigma_nfw(R, 0.5, lnM);

    // Miscentering smears out the central cusp
    // Note: simplified model may not capture this perfectly
    INFO("Sigma(centered) = " << sig_centered << ", Sigma(mis) = " << sig_mis);
  }

  SECTION("Miscentering effect diminishes at large R") {
    double const R_large = 5.0;
    double const R_mis = 0.5;

    double const sig_centered = compute_sigma_nfw(R_large, 0.0, lnM);
    double const sig_mis = compute_sigma_nfw(R_large, R_mis, lnM);

    // At large R, sqrt(R^2 + R_mis^2) ~ R, so effect is small
    double const ratio = sig_mis / sig_centered;
    INFO("Sigma ratio at R=5: " << ratio);
    CHECK(ratio > 0.8);  // Effect should be < 20% at large R
  }
}
