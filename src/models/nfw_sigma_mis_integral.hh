#ifndef Y3_CLUSTER_CPP_NFW_SIGMA_MIS_INTEGRAL_HH
#define Y3_CLUSTER_CPP_NFW_SIGMA_MIS_INTEGRAL_HH

#include <cmath>

// Miscentered NFW surface density by DIRECT azimuthal integration of the
// analytic NFW (Wright & Brainerd 2000), as an alternative to the precomputed
// single-miscentering lookup table (nfw_sigma_mis.hh).
//
//   Sigma_mis(R | R_mis, M) = (1/2pi) * int_0^{2pi}
//        Sigma_NFW( sqrt(R^2 + R_mis^2 - 2 R R_mis cos(phi)), M ) d(phi)
//
// The phi integrand is periodic and smooth, so the uniform midpoint rule is
// spectrally accurate: N_phi ~ 128-256 gives machine-level azimuthal
// normalisation (Sigma_mis -> Sigma_NFW as R_mis -> 0 to ~1e-5).
//
// Why this matters (issue: the two-halo mean-field "rnd" term not cancelling):
// this construction is MASS-CONSERVING for EVERY R_mis --
//   int Sigma_mis(R | R_mis) 2 pi R dR = M   for all R_mis --
// so the uniform mean field integrates to a flat sheet and cancels in the
// differential DeltaSigma. The lookup table loses ~4% of the projected mass
// at large offset (R_mis ~ 8 cMpc/h), which is exactly the regime the
// projection samples at large theta; that mass leak is what left the "rnd"
// term rising with R instead of vanishing.
//
// Units: Sigma [Msun/h / pc^2]; R, R_mis [cMpc/h]; M [Msun/h] (M_200c).
// Convention matches nfw_sigma_mis.hh / RichnessSelection nfw.py:
//   rho_crit,0 = 2.77533742639e11 Msun/h / (cMpc/h)^3 ; rho_eff = delta_c *
//   rho_crit * Omega_m ; delta_c = (200 c^3 / 3) / (ln(1+c) - c/(1+c)).
// NOTE: nfw_sigma_t.hh has delta_c MULTIPLIED by (ln(1+c)-c/(1+c)) instead of
// divided -- a bug; the correct (divided) form is used here.

namespace y3_cluster {

  class nfw_sigma_mis_integral {
  public:
    nfw_sigma_mis_integral(double c, double om, int n_phi = 256)
      : _c(c), _om(om), _n_phi(n_phi)
    {}

    // Centered analytic NFW Sigma(r, M) [Msun/h / pc^2], Wright & Brainerd 2000.
    double
    sigma_nfw(double r, double M) const
    {
      double const c = _c;
      double const rho_crit = 2.77533742639e11 * _om;   // Msun/h / (cMpc/h)^3
      double const fc = std::log(1.0 + c) - c / (1.0 + c);
      double const delta_c = (200.0 * c * c * c / 3.0) / fc;   // divide by fc
      double const r_200 = std::cbrt(3.0 * M / (800.0 * M_PI * rho_crit));
      double const r_s = r_200 / c;
      double const x = r / r_s;
      double const coeff = 2.0 * r_s * delta_c * rho_crit / 1.0e12;  // -> /pc^2

      if (x < 1.0 - 1.0e-9)
        return coeff / (x * x - 1.0) *
               (1.0 - 2.0 / std::sqrt(1.0 - x * x) *
                        std::atanh(std::sqrt((1.0 - x) / (1.0 + x))));
      if (x <= 1.0 + 1.0e-9) return coeff / 3.0;
      return coeff / (x * x - 1.0) *
             (1.0 - 2.0 / std::sqrt(x * x - 1.0) *
                      std::atan(std::sqrt((x - 1.0) / (1.0 + x))));
    }

    // Miscentered Sigma via the azimuthal (phi) integral. R_mis <= 0 -> centered.
    double
    operator()(double R, double R_mis, double M) const
    {
      if (R_mis <= 0.0) return sigma_nfw(R, M);
      double acc = 0.0;
      double const dphi = M_PI / _n_phi;   // midpoint: phi_i = (2i+1) * pi / N
      for (int i = 0; i < _n_phi; ++i) {
        double const phi = dphi * (2 * i + 1);
        double const r =
          std::sqrt(R * R + R_mis * R_mis - 2.0 * R * R_mis * std::cos(phi));
        acc += sigma_nfw(r, M);
      }
      return acc / _n_phi;
    }

  private:
    double _c;
    double _om;
    int _n_phi;
  };

}  // namespace y3_cluster
#endif
