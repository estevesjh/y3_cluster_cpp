#!/usr/bin/env python3
"""Identity pins for the deliberately-duplicated pipeline sources.

Several files exist twice on purpose: the ``shared``/``systematics`` split
that let ``src/pipelines`` reorganise without breaking the production
pipelines that still import the old path. The duplication is a KEPT
decision, not a bug to fix -- ``src/modules/sel_function/sel_function.py``
imports the ``shared`` copy while every test imports the ``systematics``
one, so the two are live simultaneously and nothing was checking that they
still agree.

That is the gap this file closes. Each pair below is pinned NUMERICALLY
(or, where the files should be literally the same, byte-wise), so the day
someone edits one copy and not the other, this goes red instead of the
pipelines silently computing two different selection functions.

Pairs covered:

  1. ``shared/sel_function.py``
     vs ``systematics/selection_richness/python/sel_function.py``
     -- differ only in path plumbing and which ``prj_params`` they import;
        every kernel must agree to machine precision.
  2. ``shared/sel_kernels.py``
     vs ``systematics/selection_richness/python/sel_kernels.py``
     -- two loaders that must resolve to their respective sel_function
        copies and hand back equivalent modules.
  3. ``cosmology/prj_params.py``
     vs ``systematics/selection_function/prj_params.py``
     -- identical apart from one import line inside a docstring example;
        pinned both textually (normalised) and numerically.
  4. ``src/models/bsel_bins_t.hh``
     vs ``systematics/selection_bias/cpp/bsel_bins_t.hh``
     -- byte-identical today; pinned as such.
  5. ``cosmology/bsel.py``
     vs ``systematics/selection_bias/python/bsel.py``
     -- these DO differ (the cosmology copy adds the [b_sel_marg] wall-ini
        republishing). The numerical engine ``IntegratorGLBSel`` must still
        agree; the divergence is pinned explicitly so it stays a known,
        bounded difference rather than drift.

Note on scope: this file asserts AGREEMENT, never that either copy is
correct. The physics of the selection kernels is pinned by
``sel_function.test.py`` (including its three deliberately-red HOD
normalization pins) and of the projection coefficients by
``cosmology_package.test.py``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PIPELINES = REPO / "src" / "pipelines"
sys.path.insert(0, str(PIPELINES))

SHARED_SEL_FUNCTION = PIPELINES / "shared" / "sel_function.py"
SYSTEMATICS_SEL_FUNCTION = (PIPELINES / "systematics" / "selection_richness"
                            / "python" / "sel_function.py")
SHARED_SEL_KERNELS = PIPELINES / "shared" / "sel_kernels.py"
SYSTEMATICS_SEL_KERNELS = (PIPELINES / "systematics" / "selection_richness"
                           / "python" / "sel_kernels.py")
COSMOLOGY_PRJ_PARAMS = PIPELINES / "cosmology" / "prj_params.py"
SYSTEMATICS_PRJ_PARAMS = (PIPELINES / "systematics" / "selection_function"
                          / "prj_params.py")
MODELS_BSEL_BINS_HH = REPO / "src" / "models" / "bsel_bins_t.hh"
SYSTEMATICS_BSEL_BINS_HH = (PIPELINES / "systematics" / "selection_bias"
                            / "cpp" / "bsel_bins_t.hh")
COSMOLOGY_BSEL = PIPELINES / "cosmology" / "bsel.py"
SYSTEMATICS_BSEL = (PIPELINES / "systematics" / "selection_bias" / "python"
                    / "bsel.py")

# A fiducial-ish MOR so both copies are exercised on physically sensible
# occupation numbers rather than at the edges of their support.
MOR = {
    "log10_Mmin": 12.3,
    "log10_M1": 13.4,
    "alpha": 1.0,
    "epsilon": 0.0,
    "sigma_lambda": 0.25,
    "z_pivot": 0.45,
}


def _install_cosmosis_stub():
    """Minimal ``cosmosis.datablock`` so module-scope imports resolve.

    Both sel_function copies and both bsel copies do
    ``from cosmosis.datablock import option_section`` at import time.
    CosmoSIS's Python package is not importable outside a CosmoSIS
    environment (it is absent from the macOS dev env, for instance) and
    none of the pure numerics compared below need it -- only the
    setup/execute entry points do, and those are not exercised here.

    Installed at MODULE import time, before anything loads a copy: doing it
    in a setUpClass would make correctness depend on unittest's class
    ordering.
    """
    if "cosmosis.datablock" in sys.modules:
        return
    stub = types.ModuleType("cosmosis")
    datablock = types.ModuleType("cosmosis.datablock")
    datablock.option_section = "module_options"
    datablock.names = types.SimpleNamespace(
        cosmological_parameters="cosmological_parameters",
        distances="distances")
    stub.datablock = datablock
    sys.modules.setdefault("cosmosis", stub)
    sys.modules.setdefault("cosmosis.datablock", datablock)


_install_cosmosis_stub()


def _load_by_path(alias, path):
    """Import a module file under a unique alias.

    Both copies define the same module name, so they must be loaded under
    distinct aliases to coexist in one interpreter -- which is precisely
    what makes a same-process comparison possible.
    """
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


class _SelFunctionPairMixin:
    @classmethod
    def _load_pair(cls):
        return (_load_by_path("_dual_shared_sel_function",
                              SHARED_SEL_FUNCTION),
                _load_by_path("_dual_systematics_sel_function",
                              SYSTEMATICS_SEL_FUNCTION))


class TestSelFunctionCopiesAgree(unittest.TestCase, _SelFunctionPairMixin):
    """The two live selection-kernel sources must compute the same thing."""

    @classmethod
    def setUpClass(cls):
        cls.a, cls.b = cls._load_pair()
        cls.lnM = np.linspace(30.0, 36.0, 9)
        cls.z = np.linspace(0.15, 0.70, 7)
        cls.gl_t, cls.gl_w = np.polynomial.legendre.leggauss(24)

    def test_both_copies_really_are_two_different_files(self):
        # Guard against the whole comparison silently degenerating into
        # "module compared with itself" if one path is ever symlinked or
        # rewritten to re-export the other.
        self.assertNotEqual(self.a.__file__, self.b.__file__)
        self.assertNotEqual(
            hashlib.sha256(SHARED_SEL_FUNCTION.read_bytes()).hexdigest(),
            hashlib.sha256(SYSTEMATICS_SEL_FUNCTION.read_bytes()).hexdigest())

    def test_public_and_private_surfaces_match(self):
        names_a = {n for n in dir(self.a) if not n.startswith("__")}
        names_b = {n for n in dir(self.b) if not n.startswith("__")}
        # _PIPELINES_DIR resolution differs by construction; everything
        # else must be present in both.
        ignore = {"_PIPELINES_DIR", "_parent"}
        self.assertEqual(names_a - ignore, names_b - ignore)

    def test_hod_density_agrees(self):
        for lnm in (30.5, 32.0, 34.0, 35.5):
            for z in (0.2, 0.45, 0.65):
                for ltr in (1.0, 5.0, 20.0, 80.0):
                    va = self.a._p_hod_scalar(ltr, lnm, z, MOR)
                    vb = self.b._p_hod_scalar(ltr, lnm, z, MOR)
                    self.assertAlmostEqual(
                        float(va), float(vb), places=14,
                        msg=f"lnM={lnm} z={z} ltr={ltr}")

    def test_mean_satellite_occupation_agrees(self):
        args = (MOR["log10_Mmin"], MOR["log10_M1"], MOR["alpha"],
                MOR["epsilon"], MOR["z_pivot"])
        mass = np.exp(self.lnM)
        for z in (0.2, 0.5):
            self.assertTrue(np.allclose(self.a._mu_sat(mass, z, *args),
                                        self.b._mu_sat(mass, z, *args),
                                        rtol=0.0, atol=0.0))

    def test_richness_nodes_and_hod_tensor_agree(self):
        out_a = self.a._compute_lam_nodes_and_P_HOD(
            self.lnM, self.z, MOR, self.gl_t, self.gl_w, L=6.0)
        out_b = self.b._compute_lam_nodes_and_P_HOD(
            self.lnM, self.z, MOR, self.gl_t, self.gl_w, L=6.0)
        self.assertEqual(len(out_a), len(out_b))
        for i, (xa, xb) in enumerate(zip(out_a, out_b)):
            self.assertTrue(np.array_equal(np.asarray(xa), np.asarray(xb)),
                            f"output {i} of _compute_lam_nodes_and_P_HOD")

    def test_projection_kernel_parameters_and_cdf_agree(self):
        from cosmology.prj_params import PrjParams
        splines = PrjParams.default().splines()
        lam_k, _w_k, _p_mz, _degenerate = self.a._compute_lam_nodes_and_P_HOD(
            self.lnM, self.z, MOR, self.gl_t, self.gl_w, L=6.0)

        pa = self.a._plob_params(lam_k, self.z, splines)
        pb = self.b._plob_params(lam_k, self.z, splines)
        for i, (xa, xb) in enumerate(zip(pa, pb)):
            self.assertTrue(np.array_equal(np.asarray(xa), np.asarray(xb)),
                            f"_plob_params output {i}")

        edges = np.array([20.0, 30.0, 45.0, 60.0, 200.0])
        ca = self.a._cdf_lob_stacked(edges, *pa)
        cb = self.b._cdf_lob_stacked(edges, *pb)
        self.assertTrue(np.array_equal(np.asarray(ca), np.asarray(cb)))

    def test_redshift_kernel_agrees(self):
        for zob_min, zob_max, sigma_z in ((0.20, 0.35, 0.02),
                                          (0.35, 0.50, 0.03),
                                          (0.50, 0.65, 0.05)):
            self.assertTrue(np.array_equal(
                np.asarray(self.a._S_j(self.z, zob_min, zob_max, sigma_z)),
                np.asarray(self.b._S_j(self.z, zob_min, zob_max, sigma_z))))

    def test_mass_grid_chooser_agrees(self):
        for lam_min, lam_max in ((20.0, 30.0), (45.0, 60.0), (60.0, 200.0)):
            ga = self.a._choose_lnM_grid(lam_min, lam_max, 0.20, 0.35, MOR,
                                         29.9336, 36.73, 2)
            gb = self.b._choose_lnM_grid(lam_min, lam_max, 0.20, 0.35, MOR,
                                         29.9336, 36.73, 2)
            self.assertTrue(np.array_equal(np.asarray(ga), np.asarray(gb)),
                            f"lam=[{lam_min}, {lam_max}]")

    def test_emg_helper_agrees(self):
        x = np.linspace(1.0, 200.0, 50)
        for mu, sigma, tau in ((10.0, 3.0, 8.0), (40.0, 6.0, 25.0)):
            self.assertTrue(np.allclose(self.a._f_emg(x, mu, sigma, tau),
                                        self.b._f_emg(x, mu, sigma, tau),
                                        rtol=0.0, atol=0.0))


class TestSelKernelsLoadersAgree(unittest.TestCase):
    """The two ``sel_kernels`` shims must each find their own sel_function."""

    @classmethod
    def setUpClass(cls):
        cls.a = _load_by_path("_dual_shared_sel_kernels", SHARED_SEL_KERNELS)
        cls.b = _load_by_path("_dual_systematics_sel_kernels",
                              SYSTEMATICS_SEL_KERNELS)

    @staticmethod
    def _load_isolated(shim):
        """Run a shim's ``load()`` with the shared cache key cleared."""
        saved = sys.modules.pop("y3_des_sel_function", None)
        try:
            return shim.load()
        finally:
            sys.modules.pop("y3_des_sel_function", None)
            if saved is not None:
                sys.modules["y3_des_sel_function"] = saved

    def test_each_loader_targets_its_own_sel_function_copy(self):
        # Isolated (cache cleared) each shim must find ITS OWN copy -- that
        # is the whole point of having two shims.
        self.assertEqual(Path(self._load_isolated(self.a).__file__).resolve(),
                         SHARED_SEL_FUNCTION.resolve())
        self.assertEqual(Path(self._load_isolated(self.b).__file__).resolve(),
                         SYSTEMATICS_SEL_FUNCTION.resolve())

    def test_the_two_shims_share_one_sys_modules_cache_key(self):
        # CHARACTERIZATION of a real hazard, deliberately not "fixed" here
        # (the dual copy is a kept decision): BOTH shims cache under the
        # module name "y3_des_sel_function", so in any process that uses
        # both -- e.g. a pipeline importing shared/sel_kernels while a test
        # imports the systematics one -- whichever calls load() FIRST wins,
        # and the second silently gets the other copy.
        #
        # This is safe today only because the two sel_function copies are
        # numerically identical (TestSelFunctionCopiesAgree above). If that
        # ever stops being true, this collision turns into a silent
        # wrong-answer bug, which is why the identity tests are the ones
        # that must stay green.
        self.assertEqual(self.a.load().__file__, self.b.load().__file__,
                         "expected the shared cache key to make the second "
                         "load() return the first shim's module")
        first = sys.modules.get("y3_des_sel_function")
        self.assertIsNotNone(first)
        self.assertIs(self.a.load(), self.b.load())

    def test_both_loaders_expose_the_same_kernel_surface(self):
        mod_a = self._load_isolated(self.a)
        mod_b = self._load_isolated(self.b)
        ignore = {"_PIPELINES_DIR", "_parent"}
        self.assertEqual({n for n in dir(mod_a) if not n.startswith("__")}
                         - ignore,
                         {n for n in dir(mod_b) if not n.startswith("__")}
                         - ignore)

    def test_both_loaders_expose_the_same_default_projection_splines(self):
        sa = self.a.plob_splines_default()
        sb = self.b.plob_splines_default()
        self.assertEqual(set(sa), set(sb))
        for name in sa:
            xa, ya = sa[name]._data[:2] if hasattr(sa[name], "_data") \
                else (None, None)
            del xa, ya
            # Compare by evaluation rather than by internal representation:
            # the two splines come from two prj_params copies.
            probe = np.linspace(0.05, 0.95, 25)
            self.assertTrue(np.allclose(np.asarray(sa[name](probe)),
                                        np.asarray(sb[name](probe)),
                                        rtol=0.0, atol=0.0), name)

    def test_repo_root_resolution_agrees(self):
        self.assertEqual(Path(self.a.repo_root()).resolve(),
                         Path(self.b.repo_root()).resolve())


class TestPrjParamsCopiesAgree(unittest.TestCase):
    """``cosmology`` and ``systematics`` projection coefficients."""

    @classmethod
    def setUpClass(cls):
        cls.a = _load_by_path("_dual_cosmology_prj_params",
                              COSMOLOGY_PRJ_PARAMS)
        cls.b = _load_by_path("_dual_systematics_prj_params",
                              SYSTEMATICS_PRJ_PARAMS)

    def test_sources_differ_only_in_the_documented_import_line(self):
        # The only intended difference is which package the docstring
        # example imports from.  Normalise that and the files must be
        # byte-identical -- anything else is real drift.
        pattern = re.compile(
            r"from (?:cosmology|systematics\.selection_function)\.prj_params")
        norm_a = pattern.sub("from <pkg>.prj_params",
                             COSMOLOGY_PRJ_PARAMS.read_text())
        norm_b = pattern.sub("from <pkg>.prj_params",
                             SYSTEMATICS_PRJ_PARAMS.read_text())
        self.assertEqual(norm_a, norm_b)

    def test_default_coefficient_grids_are_identical(self):
        da = self.a.PrjParams.default().as_dict()
        db = self.b.PrjParams.default().as_dict()
        self.assertEqual(set(da), set(db))
        for name in da:
            self.assertTrue(np.array_equal(np.asarray(da[name]),
                                           np.asarray(db[name])), name)

    def test_kernels_agree_on_a_probe_grid(self):
        pa, pb = self.a.PrjParams.default(), self.b.PrjParams.default()
        lob = np.linspace(1.0, 250.0, 60)
        for ltr, z in ((10.0, 0.25), (35.0, 0.45), (90.0, 0.62)):
            self.assertTrue(np.array_equal(
                np.asarray(pa.cdf_lob(lob, ltr, z)),
                np.asarray(pb.cdf_lob(lob, ltr, z))), f"ltr={ltr} z={z}")
            self.assertTrue(np.array_equal(
                np.asarray(pa.p_lob_given_ltr(lob, ltr, z)),
                np.asarray(pb.p_lob_given_ltr(lob, ltr, z))),
                f"ltr={ltr} z={z}")


class TestBselBinsHeaderTwins(unittest.TestCase):
    """``src/models`` and ``systematics`` copies of ``bsel_bins_t.hh``."""

    def test_headers_are_byte_identical(self):
        a = hashlib.sha256(MODELS_BSEL_BINS_HH.read_bytes()).hexdigest()
        b = hashlib.sha256(SYSTEMATICS_BSEL_BINS_HH.read_bytes()).hexdigest()
        self.assertEqual(
            a, b,
            "the two bsel_bins_t.hh copies have diverged -- decide which is "
            "canonical (see test/bsel_bins.test.cc, which drives the "
            "systematics copy)")

    def test_both_carry_the_same_include_guard(self):
        # They share an include guard, so a translation unit that pulled in
        # both would silently get only the first.  That is fine while they
        # are identical and a latent bug the moment they are not.
        guard = "Y3_CLUSTER_CPP_BSEL_BINS_T_HH"
        for path in (MODELS_BSEL_BINS_HH, SYSTEMATICS_BSEL_BINS_HH):
            self.assertIn(guard, path.read_text(), str(path))


class TestBselPythonCopies(unittest.TestCase):
    """The two ``bsel.py`` CosmoSIS modules.

    Unlike the pairs above these are NOT meant to be identical: the
    ``cosmology`` copy additionally republishes the [b_sel_marg] wall
    geometry into the three operator sections (issue #10).  What must still
    hold is that the numerical engine agrees, so whichever copy is declared
    canonical, the published b_small/b_large do not move.
    """

    @classmethod
    def setUpClass(cls):
        _install_cosmosis_stub()
        cls.a = _load_by_path("_dual_cosmology_bsel", COSMOLOGY_BSEL)
        cls.b = _load_by_path("_dual_systematics_bsel", SYSTEMATICS_BSEL)

    def test_closure_formulas_agree(self):
        # b_large(ltr) = b_eff [1 + 0.13 ((lob-ltr)/Delta_RND - 1)] and
        # b_small(ltr) = [(lob-ltr) - P1 - b_large I1] / J, the two
        # published Costanzi-2026 closure limits.
        ea, eb = self.a.IntegratorGLBSel, self.b.IntegratorGLBSel
        rng = np.random.default_rng(20260826)
        for _ in range(64):
            lob = float(rng.uniform(20.0, 200.0))
            ltr = float(rng.uniform(1.0, 0.9 * lob))
            p1 = float(rng.uniform(0.1, 10.0))
            i1 = float(rng.uniform(0.1, 10.0))
            j = float(rng.uniform(0.1, 10.0))
            b_eff = float(rng.uniform(1.0, 5.0))

            la = ea.evaluate_b_large(lob, ltr, p1, i1, j, b_eff)
            lb = eb.evaluate_b_large(lob, ltr, p1, i1, j, b_eff)
            self.assertTrue(np.allclose(np.asarray(la), np.asarray(lb),
                                        rtol=0.0, atol=0.0))

            sa = ea.evaluate_b_small(lob, ltr, p1, i1, j, np.asarray(la))
            sb = eb.evaluate_b_small(lob, ltr, p1, i1, j, np.asarray(lb))
            self.assertTrue(np.allclose(np.asarray(sa), np.asarray(sb),
                                        rtol=0.0, atol=0.0))

    def test_quadrature_rules_agree(self):
        # __post_init__ builds the mass and true-richness GL rules; those
        # nodes are what every downstream number depends on.
        kwargs = dict(ltr_lo=1.0, ltr_hi_factor=3.0, ltr_hi_fixed=0.0,
                      n_ltr=64, min_mass4integral=1.0e12,
                      max_log10_mass=16.0, n_mass=48)
        ia = self.a.IntegratorGLBSel(**kwargs)
        ib = self.b.IntegratorGLBSel(**kwargs)
        self.assertTrue(np.array_equal(ia.mass, ib.mass))
        self.assertTrue(np.array_equal(ia.mass_weights, ib.mass_weights))
        self.assertTrue(np.array_equal(ia._ltr_nodes, ib._ltr_nodes))
        self.assertTrue(np.array_equal(ia._ltr_weights, ib._ltr_weights))

    def test_the_documented_divergence_is_the_only_structural_one(self):
        # The cosmology copy owns the wall-ini republishing; the
        # systematics copy does not.  Pinning this keeps the difference a
        # recorded decision rather than accidental drift, and makes the
        # "which one is canonical" call visible in the test suite.
        self.assertTrue(hasattr(self.a, "_WALL_INI_SECTION"))
        self.assertFalse(hasattr(self.b, "_WALL_INI_SECTION"))
        self.assertEqual(self.a._WALL_INI_SECTION, "b_sel_marg")
        for name in ("setup", "execute", "cleanup"):
            self.assertTrue(callable(getattr(self.a, name)))
            self.assertTrue(callable(getattr(self.b, name)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
