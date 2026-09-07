// Selection-bias closure values on the exact observed-richness/redshift wall.
//
// The upstream b_sel_marginalised calculation produces one row for each
// (lambda_bin, zo_low, zo_high) wall bin.  The row's `lob` is the centre of
// the observed-richness interval and `zob` is the centre of the observed-
// redshift interval.  It also contains the two angular limits of the
// selection-affected halo bias, b_small and b_large.
//
// The Costanzi projection model uses these two limits to reconstruct the
// angular dependence b_sel(theta).  This reader does not interpolate in
// richness or redshift: `at` requires an exact richness-bin index and an
// exact redshift-bin centre, so a missing or duplicated wall row is an error.
#ifndef Y3_CLUSTER_CPP_BSEL_BINS_T_HH
#define Y3_CLUSTER_CPP_BSEL_BINS_T_HH

#include "utils/datablock_reader.hh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace y3_cluster::sp_detail {

  inline std::vector<double>
  read_bsel_vector(cosmosis::DataBlock& sample, char const* key)
  {
    try {
      return get_vector_double(sample, "b_sel_marginalised", key);
    } catch (std::exception const&) {
      auto const& nd = sample.view<cosmosis::ndarray<double>>(
          "b_sel_marginalised", key);
      return std::vector<double>(nd.begin(), nd.end());
    }
  }

  struct BSelBinValue {
    double lob;      // observed-richness-bin centre
    double zob;      // observed-redshift-bin centre
    double b_small;  // small-angle selection-affected bias limit
    double b_large;  // large-angle selection-affected bias limit
  };

  /**
   * Exact lookup table for the selection-affected bias closure.
   *
   * All vectors are read from the `b_sel_marginalised` datablock section and
   * must have the same nonzero length.  The constructor validates that each
   * redshift centre is the midpoint of its wall bounds.  `at` then returns
   * the closure values for one exact observed-richness/redshift wall row.
   */
  class BSelBins {
   public:
    explicit BSelBins(cosmosis::DataBlock& sample)
      : lambda_bin_(read_bsel_vector(sample, "lambda_bin"))
      , zo_low_(read_bsel_vector(sample, "zo_low"))
      , zo_high_(read_bsel_vector(sample, "zo_high"))
      , zob_(read_bsel_vector(sample, "zob"))
      , lob_(read_bsel_vector(sample, "lob"))
      , b_small_(read_bsel_vector(sample, "b_small"))
      , b_large_(read_bsel_vector(sample, "b_large"))
    {
      std::size_t const n = lambda_bin_.size();
      if (n == 0 || zo_low_.size() != n || zo_high_.size() != n ||
          zob_.size() != n || lob_.size() != n || b_small_.size() != n ||
          b_large_.size() != n) {
        throw std::runtime_error(
            "b_sel_marginalised vectors must be non-empty and aligned");
      }
      for (std::size_t i = 0; i != n; ++i) {
        if (zo_high_[i] <= zo_low_[i] ||
            std::abs(zob_[i] - 0.5 * (zo_low_[i] + zo_high_[i])) > 1.0e-12) {
          throw std::runtime_error("b_sel_marginalised redshift row is invalid");
        }
      }
    }

    /// Return the b_sel closure for one exact wall row.
    BSelBinValue at(int lambda_bin, double zob) const
    {
      int found = -1;
      for (std::size_t i = 0; i != lambda_bin_.size(); ++i) {
        if (static_cast<int>(lambda_bin_[i]) == lambda_bin &&
            std::abs(zob_[i] - zob) <= 1.0e-12) {
          if (found >= 0) {
            throw std::runtime_error("duplicate exact b_sel wall row");
          }
          found = static_cast<int>(i);
        }
      }
      if (found < 0) {
        throw std::runtime_error(
            "no exact b_sel row for lambda_bin=" + std::to_string(lambda_bin) +
            " zob=" + std::to_string(zob));
      }
      std::size_t const i = static_cast<std::size_t>(found);
      return {lob_[i], zob_[i], b_small_[i], b_large_[i]};
    }

   private:
    std::vector<double> lambda_bin_;
    std::vector<double> zo_low_, zo_high_, zob_, lob_;
    std::vector<double> b_small_, b_large_;
  };

}

#endif
