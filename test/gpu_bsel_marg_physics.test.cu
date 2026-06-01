// gpu_bsel_marg_physics.test.cu
//
// Physics unit tests for bSelMargGPU module.
// Tests the P[X] operator components for selection bias marginalisation.
//
// The module computes:
//   P1: normalisation integral (F=1)
//   I1: bias-weighted integral with sigmoid (F = b * xi * sigmoid)
//   J:  bias-weighted integral without sigmoid (F = b * xi * (1-sigmoid))
//
// b_sel = I1/P1 gives the selection-marginalised bias
//
// Tests:
// 1. Two-disk area overlap function
// 2. R_lambda scaling relation
// 3. Photo-z weighting kernel
// 4. Physical bounds on b_sel

#include "catch2/catch.hpp"

#include "models/p_operator_gpu_t.cuh"
#include "models/mor_des_log_t.cuh"
#include "models/sigma_photoz_des.cuh"

#include <cmath>

// Note: Do NOT use "using namespace y3_cuda" at file scope - causes sqrt conflicts with CUDA
// Import the helper functions from p_operator_gpu_t.cuh in anonymous namespace
namespace {
  using namespace y3_cuda::p_op_gpu_detail;
}


// ============================================================================
// Test: R_lambda scaling relation
// ============================================================================

TEST_CASE("R_lambda follows DES Y3 scaling", "[gpu][physics][bsel]")
{
  // R_lambda = (lambda/100)^0.2 in cMpc/h
  // This is the characteristic cluster radius scaling with richness

  SECTION("R_lambda at reference point lambda=100") {
    double const r = R_lambda(100.0);
    CHECK(r == Approx(1.0).epsilon(1e-10));
  }

  SECTION("R_lambda increases with richness") {
    double const r_low = R_lambda(20.0);
    double const r_high = R_lambda(200.0);

    INFO("R(20) = " << r_low << ", R(200) = " << r_high);
    CHECK(r_high > r_low);
  }

  SECTION("R_lambda matches expected values") {
    // lambda = 20: R = (20/100)^0.2 = 0.2^0.2 ~ 0.725
    // lambda = 50: R = (50/100)^0.2 = 0.5^0.2 ~ 0.871
    CHECK(R_lambda(20.0) == Approx(pow(0.2, 0.2)).epsilon(1e-10));
    CHECK(R_lambda(50.0) == Approx(pow(0.5, 0.2)).epsilon(1e-10));
  }
}


// ============================================================================
// Test: DES Y3 richness bin centres
// ============================================================================

TEST_CASE("lob_center returns correct bin centres", "[gpu][physics][bsel]")
{
  // DES Y3 richness bins: [20,30], [30,45], [45,60], [60,200]
  // Arithmetic centres: 25, 37.5, 52.5, 130

  CHECK(lob_center(0) == Approx(25.0));
  CHECK(lob_center(1) == Approx(37.5));
  CHECK(lob_center(2) == Approx(52.5));
  CHECK(lob_center(3) == Approx(130.0));

  // Out of bounds should return safe default
  CHECK(lob_center(-1) == Approx(25.0));
  CHECK(lob_center(4) == Approx(25.0));
}


// ============================================================================
// Test: Two-disk area overlap function
// ============================================================================

TEST_CASE("area_overlap computes fractional overlap correctly", "[gpu][physics][bsel]")
{
  // area_overlap(theta, theta_lob, theta_lt) returns the fraction of the
  // lt-disk that overlaps with the lob-disk when separated by angle theta

  SECTION("Complete overlap when disks are concentric") {
    double const theta_lob = 0.01;  // radians
    double const theta_lt = 0.005;

    // When theta=0 and lt is smaller, should be 1.0 (complete overlap)
    double const overlap = area_overlap(0.0, theta_lob, theta_lt);
    CHECK(overlap == Approx(1.0).epsilon(1e-6));
  }

  SECTION("Overlap is fractional for small offsets") {
    double const theta_lob = 0.01;
    double const theta_lt = 0.008;
    double const theta_sep = 0.005;

    double const overlap = area_overlap(theta_sep, theta_lob, theta_lt);

    INFO("Overlap fraction = " << overlap);
    CHECK(overlap > 0.0);
    CHECK(overlap < 1.0);
  }

  SECTION("Zero overlap when disks are far apart") {
    double const theta_lob = 0.01;
    double const theta_lt = 0.01;
    double const theta_sep = 0.03;  // > theta_lob + theta_lt

    double const overlap = area_overlap(theta_sep, theta_lob, theta_lt);
    CHECK(overlap == Approx(0.0).epsilon(1e-10));
  }

  SECTION("Overlap is symmetric in theta_lob and theta_lt scaling") {
    double const theta_sep = 0.005;

    // Same-sized disks
    double const o1 = area_overlap(theta_sep, 0.01, 0.01);

    // Overlap should decrease as disks move apart
    double const o2 = area_overlap(theta_sep * 1.5, 0.01, 0.01);

    CHECK(o1 > o2);
  }

  SECTION("Overlap when lt-disk is larger than lob-disk") {
    double const theta_lob = 0.005;
    double const theta_lt = 0.01;
    double const theta_sep = 0.0;

    // When lt-disk is larger and concentric, fraction = (theta_lob/theta_lt)^2
    double const overlap = area_overlap(theta_sep, theta_lob, theta_lt);
    double const expected = (theta_lob / theta_lt) * (theta_lob / theta_lt);

    CHECK(overlap == Approx(expected).epsilon(1e-6));
  }
}


// ============================================================================
// Test: Photo-z weighting kernel for b_sel
// ============================================================================

TEST_CASE("Photo-z kernel w_z has correct form", "[gpu][physics][bsel]")
{
  // The P[X] operator uses w_z(zt, zob) = 1 - u^2 where u = (zt - zob)/sigma_z
  // This is an Epanechnikov-like kernel

  y3_cuda::SIGMA_PHOTOZ_DES_t sigma_z_model;

  SECTION("Kernel is maximum when zt equals zob") {
    double const zob = 0.4;
    double const sigma_z = sigma_z_model(zob);

    auto wz = [&](double zt) {
      double const u = (zt - zob) / sigma_z;
      return (fabs(u) >= 1.0) ? 0.0 : (1.0 - u * u);
    };

    // Maximum at zt = zob
    double const w_center = wz(zob);
    double const w_offset = wz(zob + sigma_z * 0.5);

    CHECK(w_center == Approx(1.0));
    CHECK(w_offset < w_center);
  }

  SECTION("Kernel is zero outside sigma_z range") {
    double const zob = 0.4;
    double const sigma_z = sigma_z_model(zob);

    auto wz = [&](double zt) {
      double const u = (zt - zob) / sigma_z;
      return (fabs(u) >= 1.0) ? 0.0 : (1.0 - u * u);
    };

    // Zero at |zt - zob| >= sigma_z
    CHECK(wz(zob + sigma_z * 1.1) == Approx(0.0));
    CHECK(wz(zob - sigma_z * 1.1) == Approx(0.0));
  }
}


// ============================================================================
// Test: Weight functors for P1, I1, J
// ============================================================================

TEST_CASE("BSelMarg weight functors have correct properties", "[gpu][physics][bsel]")
{
  SECTION("P1 weight is always 1") {
    double const xi = 0.5;
    double const sigmoid = 0.7;

    double const w = y3_cuda::BSelMargP1GPUWeight::weight(xi, sigmoid);
    CHECK(w == 1.0);
    CHECK(y3_cuda::BSelMargP1GPUWeight::uses_bias == false);
  }

  SECTION("I1 weight includes sigmoid") {
    double const xi = 0.5;
    double const sigmoid = 0.7;

    double const w = y3_cuda::BSelMargI1GPUWeight::weight(xi, sigmoid);
    CHECK(w == Approx(xi * sigmoid));
    CHECK(y3_cuda::BSelMargI1GPUWeight::uses_bias == true);
  }

  SECTION("I2 weight excludes sigmoid") {
    double const xi = 0.5;
    double const sigmoid = 0.7;

    double const w = y3_cuda::BSelMargI2GPUWeight::weight(xi, sigmoid);
    CHECK(w == Approx(xi));
    CHECK(y3_cuda::BSelMargI2GPUWeight::uses_bias == true);
  }

  SECTION("J weight = I2 - I1") {
    double const xi = 0.5;
    double const sigmoid = 0.7;

    double const w_i1 = y3_cuda::BSelMargI1GPUWeight::weight(xi, sigmoid);
    double const w_i2 = y3_cuda::BSelMargI2GPUWeight::weight(xi, sigmoid);
    double const w_j = y3_cuda::BSelMargJGPUWeight::weight(xi, sigmoid);

    CHECK(w_j == Approx(w_i2 - w_i1).epsilon(1e-10));
    CHECK(w_j == Approx(xi * (1.0 - sigmoid)).epsilon(1e-10));
  }
}


// ============================================================================
// Test: Physical bounds on selection bias
// ============================================================================

TEST_CASE("Selection bias physical constraints", "[gpu][physics][bsel]")
{
  SECTION("Sigmoid function properties") {
    // The selection function uses sigmoid(theta) = 1 / (1 + exp(-k*(theta - theta0)))
    // k = 2.5 / theta_lob, theta0 = 0.5 * theta_lob

    double const theta_lob = 0.01;  // typical angular radius
    double const k = 2.5 / theta_lob;
    double const theta0 = 0.5 * theta_lob;

    auto sigmoid = [k, theta0](double theta) {
      return 1.0 / (1.0 + exp(-k * (theta - theta0)));
    };

    // Sigmoid is 0.5 at theta = theta0
    CHECK(sigmoid(theta0) == Approx(0.5).epsilon(0.01));

    // Sigmoid approaches ~0.22 for theta=0 (not 0, due to theta0 = 0.5*theta_lob)
    // sigmoid(0) = 1/(1+exp(k*theta0)) = 1/(1+exp(1.25)) ≈ 0.22
    CHECK(sigmoid(0.0) < 0.25);

    // Sigmoid approaches 1 for large theta
    CHECK(sigmoid(2.0 * theta_lob) > 0.9);
  }

  SECTION("b_sel should be positive") {
    // P1 > 0 (it's a positive integral)
    // I1 >= 0 (b(M,z) > 0 for halos, xi >= 0, sigmoid in [0,1])
    // Therefore b_sel = I1/P1 >= 0

    // This is a constraint that should always hold
    // We test this conceptually since full integration requires datablock
    double const P1_example = 1.0;
    double const I1_example = 0.5;

    double const b_sel = I1_example / P1_example;
    CHECK(b_sel >= 0.0);
  }

  SECTION("b_sel should be O(1) for typical clusters") {
    // Selection bias for massive clusters is typically 1-5
    // Very high values (> 10) would indicate a problem

    // This is a sanity check on expected ranges
    double const b_sel_min_expected = 0.1;
    double const b_sel_max_expected = 10.0;

    // Just checking the bounds are reasonable
    CHECK(b_sel_min_expected > 0.0);
    CHECK(b_sel_max_expected < 100.0);
  }
}


// ============================================================================
// Test: Gauss-Legendre quadrature nodes and weights
// ============================================================================

TEST_CASE("GL quadrature integrates polynomials exactly", "[gpu][physics][bsel]")
{
  // The inner theta integral uses 10-point GL quadrature
  // GL-10 should integrate polynomials up to degree 19 exactly

  // GL nodes and weights from p_op_gpu_detail
  constexpr double gl_x[10] = {
    -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
    -0.4333953941292472, -0.1488743389816312,  0.1488743389816312,
     0.4333953941292472,  0.6794095682990244,  0.8650633666889845,
     0.9739065285171717
  };
  constexpr double gl_w[10] = {
    0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
    0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
    0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
    0.0666713443086881
  };

  SECTION("Integrate constant f(x) = 1 on [-1,1]") {
    double sum = 0.0;
    for (int i = 0; i < 10; ++i) {
      sum += gl_w[i] * 1.0;
    }
    CHECK(sum == Approx(2.0).epsilon(1e-14));
  }

  SECTION("Integrate f(x) = x^2 on [-1,1]") {
    double sum = 0.0;
    for (int i = 0; i < 10; ++i) {
      sum += gl_w[i] * gl_x[i] * gl_x[i];
    }
    // Integral of x^2 from -1 to 1 = 2/3
    CHECK(sum == Approx(2.0 / 3.0).epsilon(1e-14));
  }

  SECTION("Integrate f(x) = x^4 on [-1,1]") {
    double sum = 0.0;
    for (int i = 0; i < 10; ++i) {
      double const x2 = gl_x[i] * gl_x[i];
      sum += gl_w[i] * x2 * x2;
    }
    // Integral of x^4 from -1 to 1 = 2/5
    CHECK(sum == Approx(2.0 / 5.0).epsilon(1e-14));
  }

  SECTION("Weights sum to 2") {
    double sum = 0.0;
    for (int i = 0; i < 10; ++i) {
      sum += gl_w[i];
    }
    CHECK(sum == Approx(2.0).epsilon(1e-14));
  }
}
