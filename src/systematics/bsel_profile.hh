// Tagged optical selection-bias correction kernels used by Shear1h2hMax.
//
// BselModels is the shared dispatcher.  The physical prescriptions remain
// separate: Costanzi26 is smooth and richness/redshift anchored, while
// Sunayama23 is piecewise and calibrated per richness-bin row; its returned
// multiplicative factor is 1 + Pi(R).
//
// Costanzi et al. (2026), arXiv:2604.05833, Appendix C, Eq. (23):
//   B_sel(R) = 1 + A x^alpha [1 + x^gamma]^((beta-alpha)/gamma),
//   x = R/R0, and R0 = R_lambda(lambda_ob)(1+z).
// Sunayama et al. (2023), arXiv:2309.13025, Eq. (28):
//   Pi(R) = Pi0 R/R0 for R <= R0, and
//   Pi(R) = Pi0 + c log(R/R0) for R > R0; B_sel(R) = 1 + Pi(R).
#ifndef Y3_CLUSTER_CPP_BSEL_PROFILE_HH
#define Y3_CLUSTER_CPP_BSEL_PROFILE_HH

#include "cosmosis/datablock/datablock.hh"
#include "pipelines/shared/lensing_helpers.hh"

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace y3_cluster {

class BselCostanzi26 {
public:
  BselCostanzi26(double A, double alpha, double beta, double gamma)
    : A_(A), alpha_(alpha), beta_(beta), gamma_(gamma)
  {
    if (!std::isfinite(A_) || !std::isfinite(alpha_) ||
        !std::isfinite(beta_) || !std::isfinite(gamma_))
      throw std::invalid_argument("BselCostanzi26 parameters must be finite");
    if (!(gamma_ > 0.0))
      throw std::invalid_argument("BselCostanzi26 gamma must be positive");
  }

  explicit BselCostanzi26(cosmosis::DataBlock& source,
                          std::string const& section = "bsel_profile_costanzi26")
    : BselCostanzi26(source.view<double>(section, "A"),
                     source.view<double>(section, "alpha"),
                     source.view<double>(section, "beta"),
                     source.view<double>(section, "gamma"))
  {}

  static double r0(double lob, double z)
  {
    if (!std::isfinite(lob) || !std::isfinite(z) ||
        !(lob > 0.0) || !(z > -1.0))
      throw std::invalid_argument("BselCostanzi26 requires lob > 0 and z > -1");
    return y3_pipelines::R_lambda(lob) * (1.0 + z);
  }

  double operator()(double R, double lob, double z) const
  {
    if (!(R >= 0.0))
      throw std::invalid_argument("BselCostanzi26 requires R >= 0");
    double const scale = r0(lob, z);
    if (R == 0.0)
      return 1.0;
    double const x = R / scale;
    return 1.0 + A_ * std::pow(x, alpha_) *
      std::pow(1.0 + std::pow(x, gamma_), (beta_ - alpha_) / gamma_);
  }

  double derivative(double R, double lob, double z) const
  {
    if (!(R > 0.0))
      throw std::invalid_argument("BselCostanzi26 derivative requires R > 0");
    double const x = R / r0(lob, z);
    double const Bm1 = (*this)(R, lob, z) - 1.0;
    return Bm1 / R * (alpha_ + beta_ * std::pow(x, gamma_)) /
           (1.0 + std::pow(x, gamma_));
  }

private:
  double A_, alpha_, beta_, gamma_;
};

class BselSunayama23 {
public:
  BselSunayama23(std::vector<double> pi0, std::vector<double> r0,
                 double c, std::vector<double> lambda_bin = {})
    : pi0_(std::move(pi0)), r0_(std::move(r0)), c_(c),
      lambda_bin_(std::move(lambda_bin))
  {
    if (pi0_.empty() || pi0_.size() != r0_.size())
      throw std::invalid_argument("BselSunayama23 pi0/r0 sizes differ");
    if (lambda_bin_.empty()) {
      lambda_bin_.resize(pi0_.size());
      for (std::size_t i = 0; i != lambda_bin_.size(); ++i)
        lambda_bin_[i] = static_cast<double>(i);
    }
    if (lambda_bin_.size() != pi0_.size())
      throw std::invalid_argument("BselSunayama23 lambda_bin size differs");
    if (!std::isfinite(c_))
      throw std::invalid_argument("BselSunayama23 c must be finite");
    for (std::size_t i = 0; i != pi0_.size(); ++i) {
      if (!std::isfinite(pi0_[i]) || !std::isfinite(r0_[i]) ||
          !(r0_[i] > 0.0) || !std::isfinite(lambda_bin_[i]) ||
          std::trunc(lambda_bin_[i]) != lambda_bin_[i] ||
          lambda_bin_[i] < std::numeric_limits<int>::min() ||
          lambda_bin_[i] > std::numeric_limits<int>::max())
        throw std::invalid_argument("BselSunayama23 parameters are invalid");
      for (std::size_t j = 0; j != i; ++j)
        if (lambda_bin_[i] == lambda_bin_[j])
          throw std::invalid_argument("BselSunayama23 lambda_bin labels must be unique");
    }
  }

  explicit BselSunayama23(cosmosis::DataBlock& source,
                          std::string const& section = "bsel_profile_sunayama23")
    : BselSunayama23(source.view<std::vector<double>>(section, "pi0"),
                     source.view<std::vector<double>>(section, "r0"),
                     source.view<double>(section, "c"),
                     source.has_val(section, "lambda_bin")
                       ? source.view<std::vector<double>>(section, "lambda_bin")
                       : std::vector<double>{})
  {}

  double operator()(double R, int bin_index) const
  {
    if (!(R >= 0.0))
      throw std::invalid_argument("BselSunayama23 requires R >= 0");
    std::size_t const row = row_for(bin_index);
    if (R <= r0_[row])
      return 1.0 + pi0_[row] * R / r0_[row];
    return 1.0 + pi0_[row] + c_ * std::log(R / r0_[row]);
  }

  double derivative(double R, int bin_index) const
  {
    if (!(R > 0.0))
      throw std::invalid_argument("BselSunayama23 derivative requires R > 0");
    std::size_t const row = row_for(bin_index);
    return R <= r0_[row] ? pi0_[row] / r0_[row] : c_ / R;
  }

private:
  std::size_t row_for(int bin_index) const
  {
    for (std::size_t i = 0; i != lambda_bin_.size(); ++i)
      if (lambda_bin_[i] == static_cast<double>(bin_index))
        return i;
    throw std::out_of_range("BselSunayama23 bin is absent from lambda_bin");
  }

  std::vector<double> pi0_, r0_, lambda_bin_;
  double c_;
};

class BselModels {
public:
  enum class Tag { Costanzi26, Sunayama23 };

  static Tag parse_tag(std::string const& tag)
  {
    if (tag == "Costanzi26")
      return Tag::Costanzi26;
    if (tag == "Sunayama23")
      return Tag::Sunayama23;
    throw std::invalid_argument("BselModels tag must be Costanzi26 or Sunayama23");
  }

  BselModels(BselCostanzi26 model)
    : tag_(Tag::Costanzi26), costanzi_(std::move(model)), sunayama_(std::nullopt)
  {}

  BselModels(BselSunayama23 model)
    : tag_(Tag::Sunayama23), costanzi_(std::nullopt),
      sunayama_(std::move(model))
  {}

  static BselModels from_datablock(cosmosis::DataBlock& source,
                                   std::string const& tag,
                                   std::string const& section)
  {
    if (parse_tag(tag) == Tag::Costanzi26)
      return BselModels(BselCostanzi26(source, section));
    return BselModels(BselSunayama23(source, section));
  }

  Tag tag() const { return tag_; }

  double operator()(double R, int bin_index, double lob, double z) const
  {
    if (tag_ == Tag::Costanzi26)
      return costanzi_->operator()(R, lob, z);
    return (*sunayama_)(R, bin_index);
  }

  double derivative(double R, int bin_index, double lob, double z) const
  {
    if (tag_ == Tag::Costanzi26)
      return costanzi_->derivative(R, lob, z);
    return sunayama_->derivative(R, bin_index);
  }

private:
  Tag tag_;
  std::optional<BselCostanzi26> costanzi_;
  std::optional<BselSunayama23> sunayama_;
};

} // namespace y3_cluster

#endif
