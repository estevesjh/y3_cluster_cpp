#ifndef Y3_CLUSTER_MAKE_INTERP_2D_HH
#define Y3_CLUSTER_MAKE_INTERP_2D_HH

#include "cosmosis/datablock/datablock.hh"
#include "cosmosis/datablock/ndarray.hh"
#include "utils/interp_2d.hh"

#include <vector>

namespace y3_cluster {

  using doubles = std::vector<double>;

  inline Interp2D
  make_Interp2D(cosmosis::DataBlock& cfg,
                char const* section,
                char const* x_axis_name,
                char const* y_axis_name,
                char const* z_array_name,
                bool bicubic = false)
  {
    return Interp2D(cfg.view<doubles>(section, x_axis_name),
                    cfg.view<doubles>(section, y_axis_name),
                    cfg.view<cosmosis::ndarray<double>>(section, z_array_name),
                    bicubic);
  }

  inline Interp2D
  make_Interp2D(cosmosis::DataBlock& cfg,
                char const* s1,
                char const* x_axis_name,
                char const* s2,
                char const* y_axis_name,
                char const* s3,
                char const* z_array_name)
  {
    return Interp2D(cfg.view<doubles>(s1, x_axis_name),
                    cfg.view<doubles>(s2, y_axis_name),
                    cfg.view<cosmosis::ndarray<double>>(s3, z_array_name));
  }
}

#endif
