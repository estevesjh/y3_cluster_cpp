#ifndef Y3_CLUSTER_INTERP_2D_HH
#define Y3_CLUSTER_INTERP_2D_HH

#include "point_3d.hh"

#include "cosmosis/datablock/ndarray.hh"
#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

// Interp2D is used for bilinear interpolation in 2 dimensions.
// It uses the GSL library to do the actual interpolation.
// Interp2D object do not allow extrapolation.
//
// bicubic = true uses GSL bicubic instead of bilinear interpolation (needs at
// least 4 points per axis); it removes the derivative kinks at every knot
// that otherwise stall quadrature convergence over the table.
//
namespace y3_cluster {

  // This is based on std::clamp<T> from C++ 17, put here for use with
  // models that use interpolation and need to avoid extrapolation.
  constexpr double
  do_clamp(double v, double lo, double hi)
  {
    assert(!(hi < lo));
    return (v < lo) ? lo : (hi < v) ? hi : v;
  }

  class Interp2D {
    using gsl_interp_ptr = void*;

  public:
    template <std::size_t M, std::size_t N>
    using matrix = std::array<std::array<double, N>, M>;

    // Interpolator created from three arrays; compiler assures the z-array
    // lengths match those of x- and y-arrays.
    template <std::size_t M, std::size_t N>
    Interp2D(std::array<double, M> const& xs,
             std::array<double, N> const& ys,
             matrix<M, N> const& zs);

    // Interpolator created from 3 vectors, specifying the x-axis, the y-axis,
    // and the column-major (N.B: not the natural-for-C++ row-major) storage of
    // the z values. We require the column-major ordering
    // because that is what is used by GSL.
    Interp2D(std::vector<double> const& xs,
             std::vector<double> const& ys,
             std::vector<double> const& zs,
             bool bicubic = false);

    // Interpolator created from two vectors and one vector-of-vectors.
    Interp2D(std::vector<double> const& xs,
             std::vector<double> const& ys,
             std::vector<std::vector<double>> const& zs);

    // Like above - assumes ndarray is 2D and fits it appropriately
    Interp2D(std::vector<double> const& xs,
             std::vector<double> const& ys,
             cosmosis::ndarray<double> const& zs,
             bool bicubic = false);

    // Interpolator created from vector of triplets: throws std::logic_error if
    // the points do not lie on a grid in (x,y) space, or if any values are NaN
    // or infinities. Any denormalized x- or y-values are flushed to zero.
    Interp2D(std::vector<Point3D> const& data);

    Interp2D(Interp2D const& other);
    Interp2D& operator=(Interp2D const& rhs);
    Interp2D(Interp2D&&) noexcept;
    Interp2D& operator=(Interp2D&& rhs) noexcept;

    // Destructor must clean up allocated GSL resources.
    ~Interp2D() noexcept;

    void swap(Interp2D& other) noexcept;

    double operator()(double x, double y) const;

    double
    eval(double x, double y) const
    {
      return this->operator()(x, y);
    };

    double
    clamp(double x, double y) const
    {
      return eval(do_clamp(x, min_x(), max_x()), do_clamp(y, min_y(), max_y()));
    }

    // Return the number of grid points in x and y.
    std::size_t nx() const;
    std::size_t ny() const;

  private:
    std::vector<double> xs_;
    std::vector<double> ys_;
    std::vector<double> zs_; // stores z-values in row-major
    gsl_interp_ptr interp_;
    bool bicubic_ = false;

    void init_gsl_bits_();

    // Discover the (x,y) grid implicit in the supplied set of points, or throw
    // a std::domain_error if no grid can be constructed from these points.
    void make_grid_(std::vector<Point3D> const& data);

    // Get the limits of x and y.
    double
    min_x() const
    {
      return xs_.front();
    }
    double
    max_x() const
    {
      return xs_.back();
    }
    double
    min_y() const
    {
      return ys_.front();
    }
    double
    max_y() const
    {
      return ys_.back();
    }
  };
} // namespace y3_cluster

template <std::size_t M, std::size_t N>
inline y3_cluster::Interp2D::Interp2D(std::array<double, M> const& xs,
                                      std::array<double, N> const& ys,
                                      matrix<M, N> const& zs)
  : xs_(cbegin(xs), cend(xs)), ys_(cbegin(ys), cend(ys)), zs_(M * N)
{
  for (std::size_t i = 0; i != M; ++i) {
    std::array<double, N> const& row = zs[i];
    for (std::size_t j = 0; j != N; ++j) { zs_[i + j * M] = row[j]; }
  }
  init_gsl_bits_();
}

inline void
y3_cluster::Interp2D::swap(Interp2D& other) noexcept
{
  using std::swap;
  swap(xs_, other.xs_);
  swap(ys_, other.ys_);
  swap(zs_, other.zs_);
  swap(interp_, other.interp_);
  swap(bicubic_, other.bicubic_);
}

inline std::size_t
y3_cluster::Interp2D::nx() const
{
  return xs_.size();
}

inline std::size_t
y3_cluster::Interp2D::ny() const
{
  return ys_.size();
}
#endif
