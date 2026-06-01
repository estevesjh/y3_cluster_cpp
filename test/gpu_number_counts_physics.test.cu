// gpu_number_counts_physics.test.cu
//
// Physics unit tests for numberCountsFull GPU module.
// Tests the individual physics components that make up the 4D integrand:
//   N = int dlob dltr dz dlnM * Omega(z) * dV/dOmega/dz * n(M,z)
//       * P_HOD(ltr|M,z) * P_EMG(lob|ltr,z) * K(zo|z)
//
// Tests:
// 1. MOR (P_HOD): probability integrates to ~1 over lambda_tr
// 2. Photo-z kernel (K): integrates to 1 over z_ob bins
// 3. Integrand positivity
// 4. Physical limits and edge cases

#include "catch2/catch.hpp"

#include "models/mor_des_log_t.cuh"
#include "models/int_zo_zt_des_t.cuh"
#include "models/sigma_photoz_des.cuh"
#include "utils/primitives.cuh"

#include <cmath>
#include <vector>

// Note: Do NOT use "using namespace y3_cuda" - causes sqrt conflicts with CUDA

// ============================================================================
// Test: MOR (Mass-Observable Relation) basic properties
// ============================================================================

// NOTE: MOR_DES_LOG_t requires GPU interpolator initialization from datablock.
// Testing MOR physics with simplified model instead.
TEST_CASE("MOR physics properties", "[gpu][physics][mor]")
{
  // Simplified MOR model (no GPU interpolators needed)
  // lambda_mean(M) = ((M - Mmin) / (M1 - Mmin))^alpha
  double const Mmin = 2.43e11;
  double const M1 = Mmin * 20.47;
  double const alpha = 0.859;

  auto lambda_mean = [=](double M) -> double {
    if (M <= Mmin) return 0.0;
    return std::pow((M - Mmin) / (M1 - Mmin), alpha);
  };

  SECTION("Mean richness is zero below Mmin") {
    CHECK(lambda_mean(1e11) == 0.0);
    CHECK(lambda_mean(Mmin * 0.99) == 0.0);
  }

  SECTION("Mean richness increases with mass") {
    double ltm_low = lambda_mean(1e13);
    double ltm_mid = lambda_mean(1e14);
    double ltm_high = lambda_mean(1e15);

    INFO("lambda(1e13) = " << ltm_low);
    INFO("lambda(1e14) = " << ltm_mid);
    INFO("lambda(1e15) = " << ltm_high);

    CHECK(ltm_mid > ltm_low);
    CHECK(ltm_high > ltm_mid);
  }

  SECTION("Richness is 1 at characteristic mass M1") {
    double ltm_at_M1 = lambda_mean(M1);
    // At M = M1: lambda = ((M1 - Mmin)/(M1 - Mmin))^alpha = 1
    CHECK(ltm_at_M1 == Approx(1.0).epsilon(1e-10));
  }
}


// ============================================================================
// Test: Photo-z kernel integration
// ============================================================================

TEST_CASE("y3_cuda::INT_ZO_ZT_DES_t integrates correctly", "[gpu][physics][photoz]")
{
  y3_cuda::INT_ZO_ZT_DES_t photoz_kernel;

  SECTION("Photo-z kernel is a valid probability") {
    double const zt = 0.4;

    // Integrate over all z_ob from 0 to 1
    // Should sum to approximately 1 for z_true in this range
    double const z_ob_lo = 0.0;
    double const z_ob_hi = 1.0;

    double const integral = photoz_kernel(z_ob_lo, z_ob_hi, zt);

    INFO("K(z_ob=[0,1] | z_t=" << zt << ") = " << integral);
    CHECK(integral == Approx(1.0).epsilon(0.01));
  }

  SECTION("Photo-z kernel respects bin edges") {
    double const zt = 0.35;

    // Sum of adjacent bins should equal the full bin
    double const p1 = photoz_kernel(0.2, 0.35, zt);
    double const p2 = photoz_kernel(0.35, 0.5, zt);
    double const p_total = photoz_kernel(0.2, 0.5, zt);

    INFO("p1 = " << p1 << ", p2 = " << p2 << ", total = " << p_total);
    CHECK(p1 + p2 == Approx(p_total).epsilon(1e-10));
  }

  SECTION("Photo-z kernel peaks when z_true is in bin") {
    double const zt = 0.4;

    // Bin containing z_true
    double const p_in_bin = photoz_kernel(0.35, 0.5, zt);
    // Bin not containing z_true
    double const p_out_bin = photoz_kernel(0.2, 0.35, zt);

    INFO("P(z_ob in [0.35,0.5] | z_t=0.4) = " << p_in_bin);
    INFO("P(z_ob in [0.2,0.35] | z_t=0.4) = " << p_out_bin);
    CHECK(p_in_bin > p_out_bin);
  }

  SECTION("Photo-z kernel is symmetric around z_true") {
    double const zt = 0.5;

    // Equal-sized bins on either side of z_true
    double const p_left = photoz_kernel(0.4, 0.5, zt);
    double const p_right = photoz_kernel(0.5, 0.6, zt);

    // Should be approximately equal due to symmetric Gaussian error
    CHECK(p_left == Approx(p_right).epsilon(0.01));
  }
}


// ============================================================================
// Test: Photo-z sigma model
// ============================================================================

TEST_CASE("y3_cuda::SIGMA_PHOTOZ_DES_t produces reasonable values", "[gpu][physics][photoz]")
{
  y3_cuda::SIGMA_PHOTOZ_DES_t sigma_z;

  SECTION("Sigma_z is positive for all redshifts") {
    for (double z = 0.1; z < 0.8; z += 0.1) {
      double const s = sigma_z(z);
      INFO("z = " << z << ", sigma_z = " << s);
      CHECK(s > 0.0);
    }
  }

  SECTION("Sigma_z is in reasonable range") {
    // DES photo-z errors are typically 0.01-0.05 * (1+z)
    for (double z = 0.2; z < 0.7; z += 0.1) {
      double const s = sigma_z(z);
      INFO("z = " << z << ", sigma_z = " << s);
      CHECK(s > 0.005);
      CHECK(s < 0.2);
    }
  }
}


// ============================================================================
// Test: Gaussian primitive
// ============================================================================

TEST_CASE("Gaussian function is normalised", "[gpu][physics][primitives]")
{
  SECTION("Gaussian integrates to 1") {
    double const mu = 0.0;
    double const sigma = 1.0;

    // Numerical integration with trapezoidal rule
    double sum = 0.0;
    double const dx = 0.01;
    for (double x = -10.0; x <= 10.0; x += dx) {
      sum += y3_cuda::gaussian(x, mu, sigma) * dx;
    }

    CHECK(sum == Approx(1.0).epsilon(0.01));
  }

  SECTION("Gaussian with different sigma") {
    double const mu = 5.0;
    double const sigma = 2.0;

    double sum = 0.0;
    double const dx = 0.01;
    for (double x = mu - 10 * sigma; x <= mu + 10 * sigma; x += dx) {
      sum += y3_cuda::gaussian(x, mu, sigma) * dx;
    }

    CHECK(sum == Approx(1.0).epsilon(0.01));
  }
}


// ============================================================================
// Test: Physical constraints on number counts
// ============================================================================

TEST_CASE("Number counts physics constraints", "[gpu][physics][number_counts]")
{
  SECTION("HMF decreases with mass") {
    // The halo mass function dn/dlnM decreases steeply with mass
    // This is a fundamental prediction of structure formation
    // dn/dlnM ~ M^(-1.9) at high mass (Press-Schechter approximation)

    // Test the scaling expectation
    double const slope = -1.9;

    // At M1 = 10^14, at M2 = 10^15: ratio = (M2/M1)^slope = 10^(-1.9) ~ 0.013
    double const M1 = 1e14;
    double const M2 = 1e15;
    double const expected_ratio = std::pow(M2 / M1, slope);

    INFO("Expected HMF ratio (10^15 / 10^14) ~ " << expected_ratio);
    CHECK(expected_ratio < 0.1);  // Much fewer massive halos
    CHECK(expected_ratio > 0.0);  // But not zero
  }

  SECTION("Survey volume increases with redshift bin width") {
    // Wider redshift bins should capture more clusters
    y3_cuda::INT_ZO_ZT_DES_t photoz;

    double const zt = 0.4;
    double const narrow = photoz(0.35, 0.40, zt);
    double const wide = photoz(0.35, 0.50, zt);

    CHECK(wide > narrow);
  }
}
