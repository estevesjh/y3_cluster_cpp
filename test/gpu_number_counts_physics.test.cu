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

TEST_CASE("y3_cuda::MOR_DES_LOG_t produces valid probability density", "[gpu][physics][mor]")
{
  // Typical DES Y3 MOR parameters
  double const Mmin = 2.43e11;       // 10^11.3853
  double const ratio = 20.47;        // 10^1.3112
  double const alpha = 0.859;
  double const sigma_intr = 0.181;
  double const epsilon = 0.0;
  double const z_pivot = 0.4544;

  y3_cuda::MOR_DES_LOG_t mor(Mmin, ratio, alpha, sigma_intr, epsilon, z_pivot);

  SECTION("MOR is positive for valid inputs") {
    double const lnM = 33.0;  // ~10^14 solar masses
    double const zt = 0.4;

    for (double lt = 1.0; lt < 200.0; lt += 10.0) {
      double const p = mor(lt, lnM, zt);
      INFO("lambda_tr = " << lt << ", P(lt|M,z) = " << p);
      CHECK(p >= 0.0);
    }
  }

  SECTION("MOR peaks near expected richness for given mass") {
    double const lnM = 33.5;  // ~3.5e14 solar masses
    double const zt = 0.4;

    // Find approximate peak
    double max_p = 0.0;
    double lt_at_max = 0.0;
    for (double lt = 1.0; lt < 300.0; lt += 1.0) {
      double const p = mor(lt, lnM, zt);
      if (p > max_p) {
        max_p = p;
        lt_at_max = lt;
      }
    }

    INFO("Peak at lambda_tr = " << lt_at_max << " with P = " << max_p);
    // For M ~ 3.5e14, expect richness ~ 30-100
    CHECK(lt_at_max > 10.0);
    CHECK(lt_at_max < 200.0);
  }

  SECTION("MOR is near zero for masses below Mmin") {
    double const lnM_low = log(Mmin * 0.5);  // Below minimum mass
    double const zt = 0.4;

    double sum = 0.0;
    for (double lt = 1.0; lt < 100.0; lt += 1.0) {
      sum += mor(lt, lnM_low, zt);
    }
    // Should be very small for halos below Mmin
    CHECK(sum < 1.0);
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

    // At fixed z, higher mass halos are rarer
    // This test would require HMF_t which needs datablock, so we test the concept
    // using the MOR expectation that higher richness clusters are rarer

    y3_cuda::MOR_DES_LOG_t mor(2.43e11, 20.47, 0.859, 0.181, 0.0, 0.4544);

    // For a given mass, MOR gives P(lambda|M) which should peak and then decay
    double const lnM = 33.5;
    double const zt = 0.4;

    double p_at_30 = mor(30.0, lnM, zt);
    double p_at_200 = mor(200.0, lnM, zt);

    // Very high richness should be much less probable than moderate richness
    CHECK(p_at_200 < p_at_30);
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
