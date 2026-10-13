#include "interp_2d.hh"
#include "fpsupport.hh"

#include "gsl/gsl_interp2d.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
  inline void
  free_if_nonnull(void* p)
  {
    if (p == nullptr) return;
    gsl_interp2d* interp = static_cast<gsl_interp2d*>(p);
    gsl_interp2d_free(interp);
  }
}

y3_cluster::Interp2D::Interp2D(std::vector<Point3D> const& data)
  : xs_(), ys_(), zs_(), interp_(nullptr)
{
  make_grid_(data);
  init_gsl_bits_();
}

y3_cluster::Interp2D::Interp2D(std::vector<double> const& xs,
                               std::vector<double> const& ys,
                               cosmosis::ndarray<double> const& zs,
                               bool bicubic)
  : Interp2D(xs, ys, std::vector<double>{zs.begin(), zs.end()}, bicubic)
{
  if ((zs.extents()[1] != xs.size()) || (zs.extents()[0] != ys.size())) {
    std::cerr << "Interp2D -- wrong input dimensions:\n\t"
              << "xs.size() = " << xs.size() << "\n\t"
              << "ys.size() = " << ys.size() << "\n\t"
              << "zs.shape[1] = " << zs.extents()[1] << "\n\t"
              << "zs.shape[0] = " << zs.extents()[0] << "\n";
    throw std::domain_error("Interp2D -- ndarray wrong dimensions");
  }
}

y3_cluster::Interp2D::Interp2D(std::vector<double> const& xs,
                               std::vector<double> const& ys,
                               std::vector<double> const& zs,
                               bool bicubic)
  : xs_(xs), ys_(ys), zs_(zs), bicubic_(bicubic)
{
  if (zs_.size() != nx() * ny())
    throw std::domain_error("Interp2D -- wrong number of z values passed");
  init_gsl_bits_();
}

y3_cluster::Interp2D::Interp2D(std::vector<double> const& xs,
                               std::vector<double> const& ys,
                               std::vector<std::vector<double>> const& zs)
  : xs_(xs), ys_(ys), zs_(xs.size() * ys.size())
{
  if (zs.size() != xs.size())
    throw std::domain_error("Interp2D -- wrong number of rows in z values");

  for (std::size_t i = 0; i < xs.size(); ++i) {
    std::vector<double> const& row = zs[i];
    if (row.size() != ys.size())
      throw std::domain_error(
        "Interp2D -- wrong number of columns in z values");
    for (std::size_t j = 0; j < ys.size(); ++j) {
      zs_[i + j * ys.size()] = row[j];
    }
  }

  if (zs_.size() != nx() * ny())
    throw std::domain_error("Interp2D -- wrong number of z values passed");
  init_gsl_bits_();
}

y3_cluster::Interp2D::Interp2D(Interp2D const& other)
  : xs_(other.xs_)
  , ys_(other.ys_)
  , zs_(other.zs_)
  , interp_(nullptr)
  , bicubic_(other.bicubic_)
{
  init_gsl_bits_();
}

y3_cluster::Interp2D&
y3_cluster::Interp2D::operator=(Interp2D const& rhs)
{
  Interp2D tmp(rhs);
  swap(tmp);
  return *this;
}

y3_cluster::Interp2D::Interp2D(Interp2D&& other) noexcept
  : xs_(std::move(other.xs_))
  , ys_(std::move(other.ys_))
  , zs_(std::move(other.zs_))
  , interp_(other.interp_)
  , bicubic_(other.bicubic_)
{
  other.interp_ = nullptr;
}

y3_cluster::Interp2D&
y3_cluster::Interp2D::operator=(Interp2D&& rhs) noexcept
{
  if (this == &rhs) return *this;
  free_if_nonnull(interp_);
  xs_ = std::move(rhs.xs_);
  ys_ = std::move(rhs.ys_);
  zs_ = std::move(rhs.zs_);
  interp_ = rhs.interp_;
  bicubic_ = rhs.bicubic_;
  rhs.interp_ = nullptr;
  return *this;
}

y3_cluster::Interp2D::~Interp2D() noexcept
{
  free_if_nonnull(interp_);
}

double
y3_cluster::Interp2D::operator()(double x, double y) const
{
  // We do not use the accelerator features of GSL interpolation, because we
  // do not expect that the pattern of calls will be such that it will help.
  // Profile the resulting integration routine to see if this should be changed.
  double z = 0.0;
  gsl_interp2d* temp = static_cast<gsl_interp2d*>(interp_);
  int rc = gsl_interp2d_eval_e(
    temp, xs_.data(), ys_.data(), zs_.data(), x, y, nullptr, nullptr, &z);
  if (rc != 0) throw std::domain_error("argument out of range in Interp1D");
  return z;
}

void
y3_cluster::Interp2D::init_gsl_bits_()
{
  gsl_interp2d* temp = gsl_interp2d_alloc(
    bicubic_ ? gsl_interp2d_bicubic : gsl_interp2d_bilinear, nx(), ny());
  gsl_interp2d_init(temp, xs_.data(), ys_.data(), zs_.data(), nx(), ny());
  interp_ = temp;
}

void
y3_cluster::Interp2D::make_grid_(std::vector<Point3D> const& data)
{
  if (data.empty()) throw std::domain_error("Interp2D -- no points provided");
  // Discover the (x,y) grid implied by the points we are given -- or reject
  // them as not a grid. Points in x are equivalent if they differ by an
  // absolute value less than xfuzz, and similarly for points in y.
  auto clean_data = data;
  assure_clean_floats(clean_data);

  // Sort first by y, then x, then z, within equivalence classes determined
  // by given relative and absolute tolerances. This is so that we have the
  // GSL-expected column-major ordering of the points.
  // TODO: Determine what tolerances are actually useful, and how they should
  // be exposed in the interface.
  double const reltol = 1.e-6;
  double const abstol = 1.e-24;
  Point3DLess comp{reltol, abstol, reltol, abstol};
  std::sort(clean_data.begin(), clean_data.end(), comp);

  // determine length of first block of points (how many x values for that y).
  // This is determining the value of M, the length of a column, which is the
  // number of rows.
  auto equal = [reltol, abstol, pa = clean_data.begin()](Point3D const& b) {
    return fpsupport::is_equivalent((*pa)[1], b[1], reltol, abstol);
  };

  auto const column_1_end =
    std::find_if_not(clean_data.begin(), clean_data.end(), equal);
  if (column_1_end == clean_data.end())
    throw std::domain_error("Interp2D -- Only one column");

  auto const candidate_nrows = std::distance(clean_data.begin(), column_1_end);
  xs_.resize(candidate_nrows);

  // Capture the x values; if the constructor completes, these will have been
  // the correct values.
  for (std::size_t i = 0; i != static_cast<std::size_t>(candidate_nrows); ++i) {
    xs_[i] = clean_data[i][0];
  }

  // Make sure the number of points is appropriate; determine the number of
  // columns.
  auto const candidate_ncolumns = clean_data.size() / candidate_nrows;
  if (candidate_ncolumns * candidate_nrows != clean_data.size())
    throw std::domain_error("Interp2D -- Not an integral number of columns");

  ys_.resize(candidate_ncolumns);

  // Make sure remaining columns conform: right number of rows, matching
  // x values. Capture the y values.
  auto const p_firstcolumn = clean_data.cbegin();
  ys_[0] = (*p_firstcolumn)[1];

  auto equivalent_x_values = [reltol, abstol](Point3D const& a,
                                              Point3D const& b) {
    return fpsupport::is_equivalent(a[0], b[0], reltol, abstol);
  };
  for (std::size_t column = 1; column != candidate_ncolumns; ++column) {
    auto res = std::mismatch(p_firstcolumn,
                             p_firstcolumn + candidate_nrows,
                             p_firstcolumn + column * candidate_nrows,
                             equivalent_x_values);
    if (res.first != p_firstcolumn + candidate_nrows)
      throw std::domain_error("Interp2D -- Points do not form a grid");
    ys_[column] = (*(p_firstcolumn + column * candidate_nrows))[1]; // ugh
  }
  // Capture the z-matrix. Since the points are sorted, the order of the points
  // is the correct order for the zs_ vector.
  zs_.reserve(candidate_nrows * candidate_ncolumns);
  for (auto const& p : clean_data) zs_.push_back(p[2]);
}
