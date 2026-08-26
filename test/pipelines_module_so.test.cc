// Load-time smoke test for every CPU pipelines CosmoSIS module.
//
// Each src/pipelines backend ships as a MODULE library whose only public
// surface is the C ABI that DEFINE_COSMOSIS_*_MODULE emits (setup /
// execute / cleanup, see src/utils/module_macros.hh).  CosmoSIS dlopens
// that .so and dlsyms those three names; nothing in the Catch2 suite did,
// which is why every module .cc translation unit sat at 0% coverage and
// why a link-time regression (a missing symbol from a header-only model,
// an ODR clash between two instantiated templates) could reach a
// production pipeline before anything noticed.
//
// This test is deliberately shallow -- it does NOT call setup(), which
// would need a full ini's worth of options and turn this into a duplicate
// of the per-backend tests.  It checks the two things dlopen alone can
// prove and the per-backend tests structurally cannot:
//
//   * the .so links: dlopen resolves every symbol it needs;
//   * the C ABI is actually exported under the exact three names CosmoSIS
//     looks up.
//
// Module paths come from $<TARGET_FILE:...> at configure time, so this
// always tests THIS build tree (the pattern shear_prj_frozen_gpu.test.cu
// adopted after an absolute fallback path silently dlopen'd a stale
// tree's module).  CUDA modules are gated out with the rest of the CUDA
// suite; they are covered on the GPU nodes.
#include "catch2/catch.hpp"

#include <dlfcn.h>

#include <string>
#include <vector>

namespace {

  struct ModuleUnderTest {
    char const* name;
    char const* path;
  };

  // Every CPU pipelines module registered in src/modules/CMakeLists.txt.
  // The macro definitions are injected by test/CMakeLists.txt.
  std::vector<ModuleUnderTest> const MODULES{
    {"NumCountsSijGl", PIPELINES_SO_NUMCOUNTS_SIJ_GL},
    {"NumCounts3d", PIPELINES_SO_NUMCOUNTS_3D},
    {"Shear1hGl", PIPELINES_SO_SHEAR1H_GL},
    {"Shear1hRadialSeries", PIPELINES_SO_SHEAR1H_RADIAL_SERIES},
    {"Shear1h2hMax", PIPELINES_SO_SHEAR1H2H_MAX},
    {"Shear1h3d", PIPELINES_SO_SHEAR1H_3D},
    {"Shear1h2hMax3d", PIPELINES_SO_SHEAR1H2H_MAX_3D},
    {"ShearPrjGl", PIPELINES_SO_SHEAR_PRJ_GL},
    {"ShearPrjCuhre", PIPELINES_SO_SHEAR_PRJ_CUHRE},
  };

} // namespace

TEST_CASE("every CPU pipelines module dlopens and exports the CosmoSIS C ABI")
{
  for (auto const& m : MODULES) {
    INFO("module " << m.name << " at " << m.path);

    // RTLD_LOCAL so the nine modules -- which instantiate overlapping
    // header-only model templates -- do not leak symbols into each other's
    // namespace and mask a genuinely missing one.
    void* handle = dlopen(m.path, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      char const* err = dlerror();
      FAIL_CHECK("dlopen failed: " << (err ? err : "(no dlerror)"));
      continue;
    }

    for (char const* sym : {"setup", "execute", "cleanup"}) {
      dlerror(); // clear
      void* fn = dlsym(handle, sym);
      char const* err = dlerror();
      INFO("symbol " << sym);
      CHECK(fn != nullptr);
      CHECK(err == nullptr);
    }

    CHECK(dlclose(handle) == 0);
  }
}

TEST_CASE("pipelines modules are built without a lib prefix, as CosmoSIS expects")
{
  // The top-level CMakeLists strips the `lib` prefix from these MODULE
  // targets because the ini `file = .../NumCountsSijGl.so` lines name them
  // directly.  A CMake change that reinstated the prefix would break every
  // shipped ini while every unit test still passed.
  for (auto const& m : MODULES) {
    std::string const path(m.path);
    auto const slash = path.find_last_of('/');
    std::string const base =
      slash == std::string::npos ? path : path.substr(slash + 1);
    INFO("module file " << base);
    CHECK(base == std::string(m.name) + ".so");
  }
}
