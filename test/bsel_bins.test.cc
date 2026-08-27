// Unit tests for the selection-bias closure reader
// (src/pipelines/systematics/selection_bias/cpp/bsel_bins_t.hh,
// y3_cluster::sp_detail::BSelBins), which had ZERO coverage in any build --
// as does its byte-identical twin src/models/bsel_bins_t.hh.
//
// BSelBins is the hand-off point between the b_sel_marg/bsel producers and
// the whole projection branch: ShearPrjCore calls bsel_->at(lambda_bin, zob)
// once per (lob, zob) slice to get {lob, zob, b_small, b_large}, and then
// reconstructs b_sel(theta) from those two plateaus. It deliberately does
// NOT interpolate -- it demands an EXACT wall row -- so every one of its
// error paths is a real pipeline-configuration guard, not defensive noise:
// a silently-wrong row here shifts b_sel(theta) on that whole slice.
//
// The DataBlock contract (cosmosis-models/des_y3.ini: [b_sel_marg] writes
// the wall, [bsel] adds b_small/b_large, [ShearPrjGl] consumes it), all in
// section `b_sel_marginalised`, one entry per (lambda_bin, zo_low, zo_high)
// row:
//
//   lambda_bin   richness-bin index for the row      REQUIRED
//   zo_low       observed-redshift bin lower edge    REQUIRED
//   zo_high      observed-redshift bin upper edge    REQUIRED
//   zob          bin centre; MUST equal (lo+hi)/2    REQUIRED
//   lob          observed-richness-bin centre        REQUIRED
//   b_small      small-angle bias plateau            REQUIRED
//   b_large      large-angle bias plateau            REQUIRED
//
// Pinned below: the aligned-and-non-empty ctor guard, the zob-midpoint
// guard (both directions), the exact-match lookup, the duplicate-row and
// missing-row errors, the vector<double>/ndarray dual reader path, and the
// row ordering the wall convention promises (one b_small/b_large pair per
// (lambda bin, zob) row, recovered independently of storage order).
#include "catch2/catch.hpp"

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"

#include "pipelines/systematics/selection_bias/cpp/bsel_bins_t.hh"

#include <stdexcept>
#include <string>
#include <vector>

using y3_cluster::sp_detail::BSelBins;

namespace {

  // A 2 lambda-bin x 2 zo-bin wall in the row order the producers use
  // (lambda fastest within a redshift bin), matching the [b_sel_marg]
  // `lambda_bin = 0 1 2 3  0 1 2 3  0 1 2 3` convention in
  // cosmosis-models/des_y3.ini.
  struct Wall {
    std::vector<double> lambda_bin{0.0, 1.0, 0.0, 1.0};
    std::vector<double> zo_low{0.20, 0.20, 0.35, 0.35};
    std::vector<double> zo_high{0.35, 0.35, 0.50, 0.50};
    std::vector<double> zob{0.275, 0.275, 0.425, 0.425};
    std::vector<double> lob{25.0, 37.5, 25.0, 37.5};
    std::vector<double> b_small{1.10, 1.40, 1.25, 1.55};
    std::vector<double> b_large{2.10, 2.40, 2.25, 2.55};
  };

  cosmosis::DataBlock
  make_block(Wall const& w)
  {
    cosmosis::DataBlock s;
    s.put_val("b_sel_marginalised", "lambda_bin", w.lambda_bin);
    s.put_val("b_sel_marginalised", "zo_low", w.zo_low);
    s.put_val("b_sel_marginalised", "zo_high", w.zo_high);
    s.put_val("b_sel_marginalised", "zob", w.zob);
    s.put_val("b_sel_marginalised", "lob", w.lob);
    s.put_val("b_sel_marginalised", "b_small", w.b_small);
    s.put_val("b_sel_marginalised", "b_large", w.b_large);
    return s;
  }

} // namespace

TEST_CASE("BSelBins returns the exact wall row for each (lambda_bin, zob)")
{
  Wall const w;
  auto s = make_block(w);
  BSelBins const bins(s);

  for (std::size_t i = 0; i != w.lambda_bin.size(); ++i) {
    auto const v = bins.at(static_cast<int>(w.lambda_bin[i]), w.zob[i]);
    INFO("row " << i);
    CHECK(v.lob == Approx(w.lob[i]).epsilon(1e-12));
    CHECK(v.zob == Approx(w.zob[i]).epsilon(1e-12));
    CHECK(v.b_small == Approx(w.b_small[i]).epsilon(1e-12));
    CHECK(v.b_large == Approx(w.b_large[i]).epsilon(1e-12));
  }

  // Row ordering contract: the (lambda bin, zob) pair -- not the storage
  // index -- is what identifies a row, so the same four rows shuffled must
  // give the same answers. A reader that had silently keyed on position
  // would fail here.
  Wall shuffled;
  std::vector<std::size_t> const perm{3, 1, 2, 0};
  for (std::size_t k = 0; k != perm.size(); ++k) {
    std::size_t const i = perm[k];
    shuffled.lambda_bin[k] = w.lambda_bin[i];
    shuffled.zo_low[k] = w.zo_low[i];
    shuffled.zo_high[k] = w.zo_high[i];
    shuffled.zob[k] = w.zob[i];
    shuffled.lob[k] = w.lob[i];
    shuffled.b_small[k] = w.b_small[i];
    shuffled.b_large[k] = w.b_large[i];
  }
  auto s2 = make_block(shuffled);
  BSelBins const bins2(s2);
  for (std::size_t i = 0; i != w.lambda_bin.size(); ++i) {
    auto const a = bins.at(static_cast<int>(w.lambda_bin[i]), w.zob[i]);
    auto const b = bins2.at(static_cast<int>(w.lambda_bin[i]), w.zob[i]);
    CHECK(a.b_small == Approx(b.b_small).epsilon(1e-12));
    CHECK(a.b_large == Approx(b.b_large).epsilon(1e-12));
    CHECK(a.lob == Approx(b.lob).epsilon(1e-12));
  }
}

TEST_CASE("BSelBins reads the wall from an ndarray as well as a vector<double>")
{
  // The producers publish the wall either way depending on which module
  // wrote it; read_bsel_vector's catch branch is the ndarray fallback and
  // is otherwise never exercised.
  Wall const w;
  cosmosis::DataBlock s;
  auto put_nd = [&](char const* key, std::vector<double> const& v) {
    s.put_val("b_sel_marginalised", key,
              cosmosis::ndarray<double>(v, {v.size()}));
  };
  put_nd("lambda_bin", w.lambda_bin);
  put_nd("zo_low", w.zo_low);
  put_nd("zo_high", w.zo_high);
  put_nd("zob", w.zob);
  put_nd("lob", w.lob);
  put_nd("b_small", w.b_small);
  put_nd("b_large", w.b_large);

  BSelBins const bins(s);
  auto const v = bins.at(1, 0.425);
  CHECK(v.lob == Approx(37.5).epsilon(1e-12));
  CHECK(v.b_small == Approx(1.55).epsilon(1e-12));
  CHECK(v.b_large == Approx(2.55).epsilon(1e-12));
}

TEST_CASE("BSelBins ctor rejects an empty or misaligned wall")
{
  {
    Wall w;
    for (auto* v : {&w.lambda_bin, &w.zo_low, &w.zo_high, &w.zob, &w.lob,
                    &w.b_small, &w.b_large})
      v->clear();
    auto s = make_block(w);
    CHECK_THROWS_AS(BSelBins{s}, std::runtime_error);
  }

  // Each vector dropped one entry short, one at a time: a truncated
  // producer output must fail here rather than read past the short vector
  // downstream.
  for (int which = 0; which != 7; ++which) {
    Wall w;
    std::vector<double>* v[7] = {&w.lambda_bin, &w.zo_low, &w.zo_high,
                                 &w.zob, &w.lob, &w.b_small, &w.b_large};
    v[which]->pop_back();
    auto s = make_block(w);
    INFO("short vector index " << which);
    CHECK_THROWS_AS(BSelBins{s}, std::runtime_error);
  }
}

TEST_CASE("BSelBins ctor enforces zob = (zo_low + zo_high) / 2")
{
  // zob is what every consumer keys on, and the projection geometry
  // (chi_o, D_A_o, R_excl, theta_lam) is evaluated AT zob -- so a zob that
  // is not the bin centre silently mis-places the whole slice.
  {
    Wall w;
    w.zob[2] = 0.40; // true centre is 0.425
    auto s = make_block(w);
    CHECK_THROWS_AS(BSelBins{s}, std::runtime_error);
  }
  // Degenerate/inverted redshift bin.
  {
    Wall w;
    w.zo_high[1] = w.zo_low[1];
    w.zob[1] = w.zo_low[1];
    auto s = make_block(w);
    CHECK_THROWS_AS(BSelBins{s}, std::runtime_error);
  }
  {
    Wall w;
    w.zo_high[0] = 0.10; // below zo_low
    w.zob[0] = 0.5 * (w.zo_low[0] + w.zo_high[0]);
    auto s = make_block(w);
    CHECK_THROWS_AS(BSelBins{s}, std::runtime_error);
  }
  // The tolerance is 1e-12, not "close enough": a 1e-9 drift is rejected.
  {
    Wall w;
    w.zob[0] += 1.0e-9;
    auto s = make_block(w);
    CHECK_THROWS_AS(BSelBins{s}, std::runtime_error);
  }
  // ...while a sub-1e-12 float wobble is tolerated.
  {
    Wall w;
    w.zob[0] += 1.0e-14;
    auto s = make_block(w);
    CHECK_NOTHROW(BSelBins{s});
  }
}

TEST_CASE("BSelBins::at rejects missing and duplicated rows")
{
  Wall const w;
  auto s = make_block(w);
  BSelBins const bins(s);

  // No such richness bin on the wall.
  CHECK_THROWS_AS(bins.at(7, 0.275), std::runtime_error);
  // Right richness bin, redshift centre that is not a wall row -- this is
  // the case a silently-interpolating reader would get wrong.
  CHECK_THROWS_AS(bins.at(0, 0.30), std::runtime_error);
  // Exactness is 1e-12, so even a near-miss is an error rather than a
  // nearest-neighbour hit.
  CHECK_THROWS_AS(bins.at(0, 0.275 + 1.0e-9), std::runtime_error);
  CHECK_NOTHROW(bins.at(0, 0.275 + 1.0e-14));

  // Duplicated exact row: two producers writing the same wall bin, or a
  // wall built with a repeated (lambda_bin, zob) pair.
  Wall dup;
  dup.lambda_bin[2] = dup.lambda_bin[0];
  dup.zo_low[2] = dup.zo_low[0];
  dup.zo_high[2] = dup.zo_high[0];
  dup.zob[2] = dup.zob[0];
  auto s_dup = make_block(dup);
  BSelBins const bins_dup(s_dup);
  CHECK_THROWS_AS(bins_dup.at(0, 0.275), std::runtime_error);
  // The non-duplicated rows still resolve.
  CHECK_NOTHROW(bins_dup.at(1, 0.275));
}

TEST_CASE("BSelBins DataBlock contract: each required key fails loudly when absent")
{
  // The b_sel_marginalised section is written by [b_sel_marg] (the wall
  // axes) and [bsel] (b_small/b_large) -- a pipeline that runs one without
  // the other must not produce a half-populated reader.
  char const* const keys[] = {"lambda_bin", "zo_low", "zo_high", "zob",
                              "lob", "b_small", "b_large"};
  Wall const w;
  for (char const* missing : keys) {
    cosmosis::DataBlock s;
    auto put = [&](char const* key, std::vector<double> const& v) {
      if (std::string(key) != missing)
        s.put_val("b_sel_marginalised", key, v);
    };
    put("lambda_bin", w.lambda_bin);
    put("zo_low", w.zo_low);
    put("zo_high", w.zo_high);
    put("zob", w.zob);
    put("lob", w.lob);
    put("b_small", w.b_small);
    put("b_large", w.b_large);
    INFO("missing b_sel_marginalised/" << missing);
    CHECK_THROWS(BSelBins{s});
  }
}

TEST_CASE("BSelBins feeds the b_sel(theta) plateaus the projection branch expects")
{
  // Not a physics pin -- a shape contract. ShearPrjCore builds
  //   b_sel(theta) = b_small + (b_large - b_small) * sigmoid(...)
  // from exactly these two numbers, so the reader must hand back a pair
  // that is finite and ordered the way the closure defines it (the small-
  // angle plateau below the large-angle one for every Y3 wall row).
  Wall const w;
  auto s = make_block(w);
  BSelBins const bins(s);

  for (std::size_t i = 0; i != w.lambda_bin.size(); ++i) {
    auto const v = bins.at(static_cast<int>(w.lambda_bin[i]), w.zob[i]);
    CHECK(std::isfinite(v.b_small));
    CHECK(std::isfinite(v.b_large));
    CHECK(v.b_small < v.b_large);
    CHECK(v.lob > 0.0);
  }
}
