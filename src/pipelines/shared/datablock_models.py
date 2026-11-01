"""Convention-exact NumPy datavectors and model adapters.

The module has two layers. The interpolation/model classes below mirror the
shared C++ model layer. The dataclasses near the top define the explicit
CosmoSIS contracts used by the selection pipeline:

* ``HODParameters`` normalizes sampled MOR parameters;
* ``PHOD`` is the single Python implementation of the continuous HOD density
  and true richness quadrature;
* ``BSelWallVector`` represents the C++ ``P1/I1/J`` wall;
* ``BSelOutputVector`` represents one ``b_small/b_large`` pair per wall row;
* ``BSelBins`` is the exact-key consumer view used by Python shear code.

The b-selection vectors are intentionally not rectangular grids. A C++ wall
row is identified by ``(lambda_bin, zo_low, zo_high)`` and consumers resolve
that row by ``(lambda_bin, zob)`` with a floating-point equality tolerance.
There is no b-selection interpolation in this module.

Every class here mirrors one struct under ``src/models/`` (or the SelGLCore
weight builder in ``src/models/n_operator_sel_gl_t.hh``), including its
interpolation scheme, so that Python implementations under
``src/pipelines/des_y3`` compose *the same* numbers as the production
fixed-GL modules:

    HMF          <-> src/models/hmf_t.hh          (bilinear + clamp)
    DVDoDz       <-> src/models/dv_do_dz_t.hh     (linear d_a interp)
    omega_z_des  <-> src/models/omega_z_des.hh    (piecewise polynomial)
    SelStack     <-> src/models/sel_function_t.hh (bilinear, 0 outside)
    SigmaCritInv <-> lensing_weights.hh load_sigma_crit_inv (linear + clamp)
    MassZWeights <-> nosel_gl_detail::SelGLCore   (fixed-GL z contraction)

Two input adapters expose one read interface for both runtime and offline
work: ``DataBlockSource`` wraps a live ``cosmosis`` DataBlock;
``DumpSource`` wraps a ``test``-sampler ``save_dir`` dump so validation
scripts replay exactly what a real sample saw.

Convention notes replicated deliberately (do not "fix" these here):

- HMF_t's mass axis is ``ln(m_h * (omega_m - omega_nu))`` and production
  callers (SelGLCore, the P operators) query it with the *same* lnM
  coordinate used for the selection tensor. This module keeps that exact
  pairing; see the unit-conventions section of CLAUDE.md.
- The HMF nuisance factor is ``hmf_s * (log10(M) - 13.8124426028) + hmf_q``
  evaluated at the query lnM, exactly as hmf_t.hh.
- DV_DO_DZ is in (Mpc/h)^3 per steradian per unit z, with d_a in Mpc from
  the ``distances`` section and the 2997.92 Hubble-distance constant.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.special import gammaln, ndtr

GL_CACHE = {}

CLUSTER_MOR_SECTION = "cluster_mor"
BSEL_WALL_SECTION = "b_sel_marg_P1"
BSEL_OUTPUT_SECTION = "b_sel_marginalised"


def gl_nodes(a, b, n):
    """Fixed Gauss-Legendre nodes/weights on [a, b] (p_op_detail::gl_nodes)."""
    key = int(n)
    if key not in GL_CACHE:
        GL_CACHE[key] = np.polynomial.legendre.leggauss(key)
    t, w = GL_CACHE[key]
    x = 0.5 * (b - a) * t + 0.5 * (b + a)
    return x, 0.5 * (b - a) * w


# ---------------------------------------------------------------------------
# Input adapters
# ---------------------------------------------------------------------------

class DataBlockSource:
    """Small, uniform read interface for a live CosmoSIS DataBlock.

    The rest of this file deliberately does not call ``block[...]`` directly.
    Keeping all datablock access here means the same model classes can be
    replayed from a test-sampler dump through :class:`DumpSource`.

    ``array`` always returns a floating-point NumPy array. Integer keys such
    as ``lambda_bin`` are converted back to integers by the datavector that
    owns that key.
    """

    def __init__(self, block: Any):
        self.block = block

    def array(self, section: str, key: str) -> np.ndarray:
        """Read ``section/key`` as a NumPy array."""
        return np.asarray(self.block[section, key], dtype=float)

    def scalar(self, section: str, key: str) -> float:
        """Read ``section/key`` as one floating-point scalar."""
        return float(self.block[section, key])

    def has(self, section: str, key: str) -> bool:
        """Return whether ``section/key`` is present."""
        return self.block.has_value(section, key)

    def __getitem__(self, item: tuple[str, str]) -> Any:
        return self.block[item]

    def __contains__(self, item: tuple[str, str]) -> bool:
        return self.block.has_value(item[0], item[1])


class DumpSource:
    """Read adapter over a cosmosis test-sampler save_dir directory.

    Scalars live in each section's ``values.txt`` (``name = value`` rows);
    arrays are one ``<key>.txt`` per key. 3-D arrays (``s_stack``) carry a
    ``# shape = (...)`` header line that we use to reshape.
    """

    def __init__(self, dirpath: str):
        self.dir = dirpath
        self._values = {}

    def _section_dir(self, section: str) -> str:
        """Resolve a section to its on-disk directory, case-insensitively.

        The datablock is case-insensitive and writers publish mixed-case
        section names (e.g. b_sel_marg_P1), but the test sampler saves
        lowercase directories -- resolve exact first, then lowercase.
        """
        exact = os.path.join(self.dir, section)
        if os.path.isdir(exact):
            return exact
        return os.path.join(self.dir, section.lower())

    def _section_values(self, section: str) -> dict:
        """Load and cache scalar values for one dump section."""
        if section not in self._values:
            vals = {}
            path = os.path.join(self._section_dir(section), "values.txt")
            if os.path.exists(path):
                for line in open(path):
                    if "=" in line:
                        k, v = line.split("=", 1)
                        try:
                            vals[k.strip().lower()] = float(v)
                        except ValueError:
                            pass
            self._values[section] = vals
        return self._values[section]

    def array(self, section: str, key: str) -> np.ndarray:
        """Read an array file and restore its optional shape header."""
        path = os.path.join(self._section_dir(section), key.lower() + ".txt")
        shape = None
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                if "shape" in line:
                    shape = tuple(
                        int(t) for t in
                        line.split("=")[1].strip(" ()\n").split(",") if t.strip())
        out = np.loadtxt(path)
        return out.reshape(shape) if shape else out

    def scalar(self, section: str, key: str) -> float:
        """Read one scalar from ``values.txt``."""
        return self._section_values(section)[key.lower()]

    def has(self, section: str, key: str) -> bool:
        """Return whether a scalar or array key exists in the dump."""
        if key.lower() in self._section_values(section):
            return True
        return os.path.exists(
            os.path.join(self._section_dir(section), key.lower() + ".txt"))


# ---------------------------------------------------------------------------
# Shared richness/HOD and bsel datablock models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HODParameters:
    r"""Normalized parameters for the continuous shifted-Poisson HOD.

    The values correspond to the ``cluster_mor`` CosmoSIS section:

    ``log10_Mmin``
        :math:`\log_{10}` of the mass where the central occupation turns on.
    ``log10_M1``
        :math:`\log_{10}` of the satellite normalization mass.
    ``alpha`` and ``epsilon``
        Mass and redshift slopes of the satellite mean.
    ``sigma_lambda``
        Intrinsic richness-scatter parameter.
    ``z_pivot``
        Redshift pivot for the satellite evolution; defaults to ``0.45``.

    Some pipelines sample ``log10_ratio`` instead of ``log10_M1``. The
    adapter converts that ratio to ``log10_M1`` once, so every consumer uses
    the same normalized representation.
    """

    log10_Mmin: float
    log10_M1: float
    alpha: float
    epsilon: float
    sigma_lambda: float
    z_pivot: float = 0.45

    @classmethod
    def from_source(cls, source):
        """Read and normalize HOD parameters from an input adapter."""
        log10_mmin = source.scalar(CLUSTER_MOR_SECTION, "log10_Mmin")
        if source.has(CLUSTER_MOR_SECTION, "log10_ratio"):
            log10_m1 = (
                log10_mmin
                + source.scalar(CLUSTER_MOR_SECTION, "log10_ratio")
            )
        else:
            log10_m1 = source.scalar(CLUSTER_MOR_SECTION, "log10_M1")
        z_pivot = (
            source.scalar(CLUSTER_MOR_SECTION, "z_pivot")
            if source.has(CLUSTER_MOR_SECTION, "z_pivot")
            else 0.45
        )
        return cls(
            log10_Mmin=log10_mmin,
            log10_M1=log10_m1,
            alpha=source.scalar(CLUSTER_MOR_SECTION, "alpha"),
            epsilon=source.scalar(CLUSTER_MOR_SECTION, "epsilon"),
            sigma_lambda=source.scalar(CLUSTER_MOR_SECTION, "sigma_lambda"),
            z_pivot=z_pivot,
        )


@dataclass(frozen=True)
class PHOD:
    r"""Continuous HOD density shared by ``sel_function`` and ``bsel``.

    The model is the shifted continuous-Poisson form used by
    ``src/models/mor_hod_t.hh``. For a halo of mass ``M`` and redshift ``z``:

    .. math::

       \mu_{\rm sat} = \left(\frac{M-M_{\min}}{M_1-M_{\min}}\right)^\alpha
       \left(\frac{1+z}{1+z_{\rm pivot}}\right)^\epsilon,

    with the satellite mean set to zero below ``Mmin``. The central occupation
    contributes one unit of richness above ``Mmin``. The callable evaluates
    the continuous density; ``make_ltr_quadrature`` supplies the adaptive
    true richness nodes used by the fast selection calculation and by the
    b-selection mass prior.

    Parameters
    ----------
    parameters
        Normalized MOR parameters from :class:`HODParameters`.
    poisson_tol
        Below this satellite mean, use the narrow Gaussian representation of
        the zero-satellite limit.
    fallback_sigma
        Width of that narrow Gaussian representation.
    """

    parameters: HODParameters
    poisson_tol: float = 1.0e-8
    fallback_sigma: float = 1.0e-3

    @classmethod
    def from_source(cls, source):
        """Construct ``PHOD`` from ``cluster_mor`` in an input adapter."""
        return cls(HODParameters.from_source(source))

    @classmethod
    def from_datablock(cls, block):
        """Construct ``PHOD`` directly from a live CosmoSIS DataBlock."""
        return cls.from_source(DataBlockSource(block))

    def mu_sat(self, mass, z):
        """Return the mean satellite richness at ``(mass, z)``.

        ``mass`` is in linear mass units and may be any NumPy-broadcastable
        shape with ``z``. The return value has the broadcast shape. Halos below
        ``Mmin`` have zero satellite occupation.
        """
        parameters = self.parameters
        mass = np.asarray(mass, dtype=float)
        redshift = np.asarray(z, dtype=float)

        # Satellite-occupation model used by MOR_HOD_t:
        #
        #   mu_sat(M, z) =
        #       [(M - Mmin) / (M1 - Mmin)]^alpha
        #       * [(1 + z) / (1 + z_pivot)]^epsilon
        #
        # The occupation is zero for M <= Mmin.  ``mass_fraction`` is
        # clipped through ``maximum`` rather than by changing the mass grid,
        # so broadcasting and the caller's integration nodes are preserved.
        minimum_mass = 10.0 ** parameters.log10_Mmin
        satellite_mass = 10.0 ** parameters.log10_M1
        mass_interval = satellite_mass - minimum_mass
        shape = np.broadcast(mass, redshift).shape
        if mass_interval <= 0.0:
            # This is outside the allowed MOR prior, but returning zero keeps
            # the model finite and matches the C++ guard in _mu_sat().
            return np.zeros(shape, dtype=float)

        mass_above_threshold = np.maximum(mass - minimum_mass, 0.0)
        mass_fraction = np.where(
            mass_above_threshold > 0.0,
            mass_above_threshold / mass_interval,
            0.0,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                mass_fraction > 0.0,
                mass_fraction ** parameters.alpha
                * ((1.0 + redshift) / (1.0 + parameters.z_pivot))
                ** parameters.epsilon,
                0.0,
            )

    def __call__(self, lambda_true, ln_mass, z):
        r"""Evaluate :math:`P(\lambda_{true}\mid M,z)` elementwise.

        ``lambda_true`` and ``ln_mass`` are broadcast together with ``z``.
        ``ln_mass`` is the natural logarithm of linear halo mass. The result
        is a density in continuous richness, not a discrete probability mass.
        Values below the physical richness support are exactly zero.
        """
        parameters = self.parameters
        lambda_true, ln_mass, redshift = np.broadcast_arrays(
            np.asarray(lambda_true, dtype=float),
            np.asarray(ln_mass, dtype=float),
            np.asarray(z, dtype=float),
        )
        mass = np.exp(ln_mass)
        central_richness = (
            mass >= 10.0 ** parameters.log10_Mmin
        ).astype(float)
        satellite_mean = self.mu_sat(mass, redshift)

        # The intrinsic scatter changes both the Poisson mean and the shift
        # of the continuous distribution.
        scatter_shift = (parameters.sigma_lambda * satellite_mean) ** 2
        poisson_mean = satellite_mean + scatter_shift
        # x = lambda_true + delta: the "-1"/Gamma(x) of the continued
        # Poisson pmf is the central, so <lambda_true> = 1 + mu_sat above
        # Mmin (below Mmin mu_sat = 0 and the fallback applies).  Before
        # 2026-11-01 x = lambda_true - central + delta counted the central
        # twice and put <lambda_true> one unit high.
        shifted_richness = lambda_true + scatter_shift

        zero_satellite_limit = satellite_mean <= self.poisson_tol
        fallback = (
            np.exp(-0.5 * ((lambda_true - central_richness)
                           / self.fallback_sigma) ** 2)
            / (np.sqrt(2.0 * np.pi) * self.fallback_sigma)
        )

        valid_support = (
            (lambda_true >= 0.0)
            & (shifted_richness > 0.0)
        )
        safe_richness = np.where(valid_support, shifted_richness, 1.0)

        # Continuous shifted-Poisson HOD probability:
        #
        #   delta = (sigma_lambda * mu_sat)^2
        #   nu    = mu_sat + delta
        #   x     = lambda_true + delta
        #   P     = exp[-nu + (x - 1) * log(nu) - log Gamma(x)]
        #
        # ``gammaln`` evaluates log Gamma(x) directly.  This is both the
        # mathematical form used by MOR_HOD_t and more stable than computing
        # Gamma(x) first and taking its logarithm.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_pdf = (
                -poisson_mean
                + (safe_richness - 1.0)
                * np.log(np.maximum(poisson_mean, 1.0e-300))
                - gammaln(safe_richness)
            )
        density = np.where(valid_support, np.exp(log_pdf), 0.0)
        density = np.where(zero_satellite_limit, fallback, density)
        # The fallback is also restricted to physical richness values.
        return np.where(lambda_true >= 0.0, density, 0.0)

    def make_ltr_quadrature(
        self, ln_mass, z, quadrature_nodes, quadrature_weights, width=6.0
    ):
        """Build adaptive true richness quadrature arrays.

        Parameters
        ----------
        ln_mass : array-like, shape ``(N_mass,)``
            Natural-log mass grid.
        z : array-like, shape ``(N_z,)``
            Redshift grid.
        quadrature_nodes, quadrature_weights : array-like, shape ``(N_q,)``
            Standard Gauss-Legendre nodes and weights on ``[-1, 1]``.
        width : float
            Number of HOD standard deviations used on either side of the
            local mean.

        Returns
        -------
        lambda_nodes, weights, probability, degenerate
            Arrays with shapes ``(N_mass, N_z, N_q)``, ``(N_mass, N_z, N_q)``,
            ``(N_mass, N_z, N_q)``, and ``(N_mass, N_z)`` respectively.
        """
        ln_mass = np.asarray(ln_mass, dtype=float)
        z = np.asarray(z, dtype=float)
        quadrature_nodes = np.asarray(quadrature_nodes, dtype=float)
        quadrature_weights = np.asarray(quadrature_weights, dtype=float)
        mass = np.exp(ln_mass)[:, None]
        redshift = z[None, :]
        parameters = self.parameters
        satellite_mean = self.mu_sat(mass, redshift)
        central_richness = (
            mass >= 10.0 ** parameters.log10_Mmin
        ).astype(float)
        poisson_mean = (
            satellite_mean
            + (parameters.sigma_lambda * satellite_mean) ** 2
        )
        mean_richness = central_richness + satellite_mean
        standard_deviation = np.sqrt(np.maximum(poisson_mean, 0.0))
        lower_bound = np.maximum(
            0.0, mean_richness - width * standard_deviation
        )
        upper_bound = mean_richness + width * standard_deviation
        degenerate = upper_bound <= lower_bound
        half_width = 0.5 * (upper_bound - lower_bound)
        midpoint = 0.5 * (lower_bound + upper_bound)
        lambda_nodes = (
            midpoint[..., None]
            + half_width[..., None] * quadrature_nodes
        )
        weights = half_width[..., None] * quadrature_weights
        prob = self(
            lambda_nodes,
            ln_mass[:, None, None],
            z[None, :, None],
        )
        prob = np.where(degenerate[..., None], 0.0, prob)
        return lambda_nodes, weights, prob, degenerate


@dataclass(frozen=True)
class BSelWallVector:
    """The C++ inputs that ``bsel`` closes into ``B_small/B_large``.

    Every field is a one-dimensional vector with the same length
    ``N_wall``. Row ``i`` is one C++ integration point:

    ``(lambda_bin[i], zo_low[i], zo_high[i], p1[i], i1[i], j[i])``.

    ``zob`` and ``lob`` are derived values. ``zob`` is the redshift-bin
    midpoint. ``lob`` is the centre of the lambda interval selected by
    ``lambda_bin`` in ``sel_function/lambda_edges``.

    Datablock inputs
    -----------------
    ``b_sel_marg_P1/vals``
        C++ ``P1`` values. The wall keys are taken from the selection
        function's published ``zob_min/zob_max`` arrays because the generic
        C++ evaluator writes only the evaluated values.
    ``b_sel_marg_I1/vals``
        C++ ``I1`` values.
    ``b_sel_marg_J/vals``
        C++ ``J = I2 - I1`` values.
    """

    lambda_bin: np.ndarray
    zo_low: np.ndarray
    zo_high: np.ndarray
    zob: np.ndarray
    lob: np.ndarray
    p1: np.ndarray
    i1: np.ndarray
    j: np.ndarray

    @classmethod
    def from_source(cls, source, lambda_edges):
        """Read the three C++ operator vectors and derive wall centres.

        ``lambda_edges`` must be the vector published by
        ``sel_function/lambda_edges``. Keeping this argument explicit makes
        it impossible for ``bsel`` to silently reintroduce a second hard-coded
        lambda partition.
        """
        lambda_edges = np.asarray(lambda_edges, dtype=float).ravel()
        if lambda_edges.size < 2 or np.any(np.diff(lambda_edges) <= 0.0):
            raise ValueError("lambda_edges must be a strictly increasing vector")

        if source.has(BSEL_WALL_SECTION, "lambda_bin"):
            lambda_bin = source.array(
                BSEL_WALL_SECTION, "lambda_bin"
            ).astype(int).ravel()
            zo_low = source.array(BSEL_WALL_SECTION, "zo_low").ravel()
            zo_high = source.array(BSEL_WALL_SECTION, "zo_high").ravel()
        else:
            # P_operator publishes P1/I1/J values but not the evaluator's
            # input grid. The selection function owns the same 12-bin wall
            # geometry, so reconstruct its row keys from the published
            # redshift bounds and lambda partition.
            zo_low = source.array("sel_function", "zob_min").ravel()
            zo_high = source.array("sel_function", "zob_max").ravel()
            n_lambda = lambda_edges.size - 1
            if n_lambda <= 0 or zo_low.size % n_lambda != 0:
                raise ValueError(
                    "selection wall geometry is not aligned with "
                    "lambda_edges")
            lambda_bin = np.tile(
                np.arange(n_lambda, dtype=int), zo_low.size // n_lambda)
        p1 = source.array(BSEL_WALL_SECTION, "vals").ravel()
        i1 = source.array("b_sel_marg_I1", "vals").ravel()
        j = source.array("b_sel_marg_J", "vals").ravel()

        invalid_lambda_bin = (
            (lambda_bin < 0) | (lambda_bin >= lambda_edges.size - 1)
        )
        if np.any(invalid_lambda_bin):
            raise ValueError("bsel lambda_bin is outside lambda_edges")

        zob = 0.5 * (zo_low + zo_high)
        lob = 0.5 * (
            lambda_edges[lambda_bin] + lambda_edges[lambda_bin + 1]
        )
        wall = cls(lambda_bin, zo_low, zo_high, zob, lob, p1, i1, j)
        return wall.validate()

    def validate(self):
        """Check alignment, valid redshift intervals, and unique wall keys."""
        arrays = (self.lambda_bin, self.zo_low, self.zo_high, self.zob,
                  self.lob, self.p1, self.i1, self.j)
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("bsel wall fields must be one-dimensional vectors")
        sizes = {x.size for x in arrays}
        if len(sizes) != 1:
            raise ValueError("bsel wall vectors have inconsistent lengths")
        if np.any(self.zo_high <= self.zo_low):
            raise ValueError("bsel wall has invalid redshift bounds")
        if not np.allclose(self.zob, 0.5 * (self.zo_low + self.zo_high)):
            raise ValueError("bsel zob does not match the wall bounds")
        keys = list(zip(self.lambda_bin.tolist(), self.zo_low.tolist(),
                        self.zo_high.tolist()))
        if len(set(keys)) != len(keys):
            raise ValueError("bsel wall contains duplicate bins")
        return self


@dataclass(frozen=True)
class BSelBins:
    """Consumer view of the exact ``bsel`` output.

    The vectors are aligned by row. A row is selected by the same key used by
    the C++ wall: ``lambda_bin`` plus the wall redshift centre ``zob``.
    ``values`` returns ``(lob, zob, b_small, b_large)`` for that row.

    Production output contains seven one-dimensional vectors in
    ``b_sel_marginalised``. ``from_source`` also understands the old
    rectangular dump format so historical offline validation remains usable;
    that compatibility path expands values into exact rows and never performs
    interpolation.
    """

    lambda_bin: np.ndarray
    zo_low: np.ndarray
    zo_high: np.ndarray
    zob: np.ndarray
    lob: np.ndarray
    b_small: np.ndarray
    b_large: np.ndarray

    @classmethod
    def from_source(cls, source):
        """Read production vectors, or expand one historical dump format."""
        if cls._has_exact_output(source):
            return cls._read_exact_output(source)
        return cls._read_legacy_rectangular_output(source)

    @staticmethod
    def _has_exact_output(source):
        """Return whether all fields of the production contract are present."""
        exact_fields = (
            "lambda_bin", "zo_low", "zo_high", "zob", "lob",
            "b_small", "b_large",
        )
        return all(
            source.has(BSEL_OUTPUT_SECTION, field) for field in exact_fields
        )

    @classmethod
    def _read_exact_output(cls, source):
        """Read and validate the production one-row-per-wall representation."""
        fields = (
            "lambda_bin", "zo_low", "zo_high", "zob", "lob",
            "b_small", "b_large",
        )
        arrays = {
            field: source.array(BSEL_OUTPUT_SECTION, field).ravel()
            for field in fields
        }
        output = cls(arrays["lambda_bin"].astype(int), *(
            arrays[field] for field in fields[1:]
        ))
        return output.validate()

    @classmethod
    def _read_legacy_rectangular_output(cls, source):
        """Expand old ``(zob, lob)`` arrays into exact rows for replay only."""
        lob = source.array(BSEL_OUTPUT_SECTION, "lob").ravel()
        zob = source.array(BSEL_OUTPUT_SECTION, "zob").ravel()
        small = source.array(BSEL_OUTPUT_SECTION, "b_small").reshape(
            zob.size, lob.size
        )
        large = source.array(BSEL_OUTPUT_SECTION, "b_large").reshape(
            zob.size, lob.size
        )
        lambda_bin, redshift = np.meshgrid(
            np.arange(lob.size), zob
        )
        output = cls(
            lambda_bin.ravel().astype(int),
            redshift.ravel(),
            redshift.ravel(),
            redshift.ravel(),
            np.broadcast_to(lob[None, :], small.shape).ravel(),
            small.ravel(),
            large.ravel(),
        )
        return output.validate()

    def validate(self):
        """Check that all consumer vectors have identical row counts."""
        arrays = (self.lambda_bin, self.zo_low, self.zo_high, self.zob,
                  self.lob, self.b_small, self.b_large)
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("bsel consumer fields must be one-dimensional vectors")
        if len({x.size for x in arrays}) != 1:
            raise ValueError("bsel output vectors have inconsistent lengths")
        if not np.allclose(self.zob, 0.5 * (self.zo_low + self.zo_high)):
            raise ValueError("bsel zob does not match the wall")
        return self

    def find_exact_row(self, lambda_bin, zo_low=None, zo_high=None, zob=None):
        """Return one exact row as ``(lob, zob, b_small, b_large)``.

        Parameters are optional only to support the different consumers:
        production C++ and fast-mass Python normally provide ``lambda_bin``
        and ``zob``; a stricter caller may also provide ``zo_low`` and
        ``zo_high``. Exactly one row must match. A neighboring row is never
        selected when the requested centre falls between wall bins.
        """
        matches = self.lambda_bin == int(lambda_bin)
        if zo_low is not None:
            matches &= np.isclose(self.zo_low, zo_low)
        if zo_high is not None:
            matches &= np.isclose(self.zo_high, zo_high)
        if zob is not None:
            matches &= np.isclose(self.zob, zob)
        rows = np.flatnonzero(matches)
        if rows.size != 1:
            raise ValueError(
                "bsel output has no unique row for "
                f"lambda_bin={lambda_bin}, zo_low={zo_low}, "
                f"zo_high={zo_high}, zob={zob}"
            )
        i = int(rows[0])
        return self.lob[i], self.zob[i], self.b_small[i], self.b_large[i]

    def values(self, lambda_bin, zo_low=None, zo_high=None, zob=None):
        """Backward-compatible alias for :meth:`find_exact_row`."""
        return self.find_exact_row(lambda_bin, zo_low, zo_high, zob)


@dataclass(frozen=True)
class BSelOutputVector:
    """The seven vectors written by the Python ``bsel`` module.

    Each row has the form

    ``(lambda_bin, zo_low, zo_high, zob, lob, b_small, b_large)``.

    ``b_small`` and ``b_large`` are the only scale-dependent bias values
    written by the producer. Consumers reconstruct the theta dependence from
    those two values and the shared sigmoid. Keeping the wall metadata beside
    the values prevents accidental reshaping into a different bin order.
    """

    lambda_bin: np.ndarray
    zo_low: np.ndarray
    zo_high: np.ndarray
    zob: np.ndarray
    lob: np.ndarray
    b_small: np.ndarray
    b_large: np.ndarray

    def validate(self):
        """Validate one-dimensional alignment and redshift midpoints."""
        arrays = (self.lambda_bin, self.zo_low, self.zo_high, self.zob,
                  self.lob, self.b_small, self.b_large)
        size = arrays[0].size
        if any(array.ndim != 1 or array.size != size for array in arrays):
            raise ValueError("bsel output arrays must be aligned one-dimensional vectors")
        if np.any(self.zo_high <= self.zo_low):
            raise ValueError("bsel output has invalid redshift bounds")
        if not np.allclose(self.zob, 0.5 * (self.zo_low + self.zo_high)):
            raise ValueError("bsel output zob does not match the wall bounds")
        return self

    def write_to_datablock(self, block, section=BSEL_OUTPUT_SECTION):
        """Write the validated production vectors to a CosmoSIS DataBlock.

        CosmoSIS runtime arrays use the double-array convention, including
        ``lambda_bin``. Consumers cast that one key to integer before lookup.
        """
        self.validate()
        # Keep all runtime vectors in the Cosmosis double-array convention;
        # consumers cast lambda_bin to an integer key when they index the wall.
        block[section, "lambda_bin"] = np.asarray(self.lambda_bin, dtype=float)
        block[section, "zo_low"] = np.asarray(self.zo_low, dtype=float)
        block[section, "zo_high"] = np.asarray(self.zo_high, dtype=float)
        block[section, "zob"] = np.asarray(self.zob, dtype=float)
        block[section, "lob"] = np.asarray(self.lob, dtype=float)
        block[section, "b_small"] = np.asarray(self.b_small, dtype=float)
        block[section, "b_large"] = np.asarray(self.b_large, dtype=float)


# ---------------------------------------------------------------------------
# Cosmology / geometry terms
# ---------------------------------------------------------------------------

class EZ:
    """E(z) = sqrt(omega_m (1+z)^3 + omega_k (1+z)^2 + omega_lambda)."""

    def __init__(self, source):
        self.omega_m = source.scalar("cosmological_parameters", "omega_m")
        self.omega_k = source.scalar("cosmological_parameters", "omega_k")
        self.omega_lambda = source.scalar("cosmological_parameters",
                                          "omega_lambda")

    def __call__(self, z):
        zp1 = 1.0 + np.asarray(z, dtype=float)
        return np.sqrt(self.omega_m * zp1**3 + self.omega_k * zp1**2 +
                       self.omega_lambda)


class DVDoDz:
    """dV/dOmega/dz in (Mpc/h)^3 — replica of DV_DO_DZ_t."""

    def __init__(self, source):
        self._z = source.array("distances", "z")
        self._d_a = source.array("distances", "d_a")
        self._h = source.scalar("cosmological_parameters", "h0")
        self._ez = EZ(source)

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        da_h = np.interp(z, self._z, self._d_a) * self._h
        return 2997.92 * (1.0 + z)**2 * da_h * da_h / self._ez(z)


def _horner(coeffs, x):
    out = np.zeros_like(x)
    for c in coeffs:
        out = c + x * out
    return out


def omega_z_des(z):
    """Effective survey area Omega(z) in rad^2 — replica of OMEGA_Z_DES."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    c1 = [0.0, 0.0, 0.0, -0.00262353, 0.01940118, 0.45133063]
    c2 = [1.33647377e4, 1.35291046e3, -1.26204891e2,
          -2.83454918e1, -2.26465905, 3.84958753e-1]
    c3 = [0.0, 0.0, -1.88101967, 4.8071839, -4.11424324, 1.18196785]
    m1 = z < 0.504
    m2 = (~m1) & (z < 0.7)
    m3 = ~(m1 | m2)
    out[m1] = _horner(c1, z[m1])
    out[m2] = _horner(c2, z[m2] - 0.6)
    out[m3] = _horner(c3, z[m3])
    return out


# ---------------------------------------------------------------------------
# Halo mass function
# ---------------------------------------------------------------------------

class HMF:
    """dn/dlnM with the HMF_t axis and nuisance conventions.

    The x-axis is ``ln(m_h * (omega_m - omega_nu))`` and the query
    coordinate is passed through unmodified — the exact HMF_t pairing used
    by every production caller in the fixed-GL path.
    """

    def __init__(self, source):
        m_h = source.array("mass_function", "m_h")
        omega_m = source.scalar("cosmological_parameters", "omega_m")
        omega_nu = (
            source.scalar("cosmological_parameters", "omega_nu")
            if source.has("cosmological_parameters", "omega_nu")
            else 0.0
        )
        self._lnm = np.log(m_h * (omega_m - omega_nu))
        self._z = source.array("mass_function", "z")
        dndlnmh = np.asarray(source.array("mass_function", "dndlnmh"),
                             dtype=float)
        if dndlnmh.shape != (self._z.size, self._lnm.size):
            dndlnmh = dndlnmh.reshape(self._z.size, self._lnm.size)
        self._interp = RegularGridInterpolator(
            (self._z, self._lnm), dndlnmh, method="linear",
            bounds_error=False, fill_value=None)
        # The production values are normally present. These defaults preserve
        # the raw datablock HMF when a focused unit-test fixture omits the
        # optional abundance nuisance parameters.
        self._s = (
            source.scalar("cluster_abundance", "hmf_s")
            if source.has("cluster_abundance", "hmf_s")
            else 0.0
        )
        self._q = (
            source.scalar("cluster_abundance", "hmf_q")
            if source.has("cluster_abundance", "hmf_q")
            else 1.0
        )

    def __call__(self, lnM, z):
        lnM = np.asarray(lnM, dtype=float)
        z = np.asarray(z, dtype=float)
        lnM_c = np.clip(lnM, self._lnm[0], self._lnm[-1])
        z_c = np.clip(z, self._z[0], self._z[-1])
        lnM_b, z_b = np.broadcast_arrays(lnM_c, z_c)
        vals = self._interp(np.stack([z_b, lnM_b], axis=-1))
        # 0.4342944819 = log10(e); nuisance evaluated at the query lnM.
        nuis = self._s * (lnM * 0.4342944819 - 13.8124426028) + self._q
        return vals * nuis


# ---------------------------------------------------------------------------
# Selection tensor and Sigma_crit_inv
# ---------------------------------------------------------------------------

class SelStack:
    """Bilinear per-bin reader of sel_function/S_stack (SelFunction_t)."""

    def __init__(self, source):
        self.lnm = source.array("sel_function", "lnm")
        self.z = source.array("sel_function", "z")
        s_stack = np.asarray(source.array("sel_function", "s_stack"),
                             dtype=float)
        if s_stack.ndim != 3:
            s_stack = s_stack.reshape(-1, self.z.size, self.lnm.size)
        self.n_bins = s_stack.shape[0]
        self._interps = [
            RegularGridInterpolator((self.z, self.lnm), s_stack[b],
                                    method="linear", bounds_error=False,
                                    fill_value=0.0)
            for b in range(self.n_bins)
        ]

    def __call__(self, b, lnM, z):
        lnM_b, z_b = np.broadcast_arrays(np.asarray(lnM, dtype=float),
                                         np.asarray(z, dtype=float))
        return self._interps[b](np.stack([z_b, lnM_b], axis=-1))


#: Where OMEGA_Z_DES switches polynomial (OMEGA_Z_DES::breakpoints).
OMEGA_Z_DES_BREAKPOINTS = (0.504, 0.7)


def composite_z_panels(zob_min, zob_max, sigma_z, *, zt_lo, zt_hi,
                       excl=None, one_plus_z=False, step_width=5.0):
    """Panels of the composite true-z rule (models/sel_z_rule.hh).

    Breaks at every bin edge's step zob +- L sigma_z, at the seam edges
    and the OMEGA_Z_DES breakpoints, over the union of the bins' supports
    [zob_min - L sigma_z, zob_max + L sigma_z] clipped to [zt_lo, zt_hi];
    panels inside the seam ``excl = (lo, hi)`` are dropped.
    """
    zob_min = np.atleast_1d(np.asarray(zob_min, dtype=float))
    zob_max = np.atleast_1d(np.asarray(zob_max, dtype=float))
    sigma_z = np.broadcast_to(np.asarray(sigma_z, dtype=float), zob_min.shape)
    L = float(step_width)

    def width(b, z):
        return sigma_z[b] * ((1.0 + z) if one_plus_z else 1.0)

    br, lo, hi = [], float(zt_hi), float(zt_lo)
    for b in range(zob_min.size):
        wl, wh = L * width(b, zob_min[b]), L * width(b, zob_max[b])
        br += [zob_min[b] - wl, zob_min[b] + wl, zob_max[b] - wh,
               zob_max[b] + wh]
        lo, hi = min(lo, zob_min[b] - wl), max(hi, zob_max[b] + wh)
    lo, hi = max(lo, float(zt_lo)), min(hi, float(zt_hi))
    if not hi > lo:
        return []
    has_excl = excl is not None and excl[1] > excl[0]
    if has_excl:
        br += [excl[0], excl[1]]
    br += list(OMEGA_Z_DES_BREAKPOINTS) + [lo, hi]
    br = np.sort(np.clip(br, lo, hi))
    out = []
    for a, c in zip(br[:-1], br[1:]):
        if not c - a > 1e-12:
            continue
        if has_excl and a >= excl[0] - 1e-12 and c <= excl[1] + 1e-12:
            continue
        out.append((float(a), float(c)))
    return out


def composite_gl_nodes(panels, n_panel):
    """Concatenated GL nodes/weights, ``n_panel`` per panel."""
    if not panels:
        return np.empty(0), np.empty(0)
    xs, ws = zip(*(gl_nodes(a, c, n_panel) for a, c in panels))
    return np.concatenate(xs), np.concatenate(ws)


class NaturalSplineGrid:
    """Natural cubic spline on knots ``x`` for many rows at once -- the
    numpy twin of sel_z_detail::NaturalSplineGrid (models/sel_z_rule.hh):
    same Thomas factorisation, same operation order."""

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        n = self.x.size
        if n < 3:
            raise ValueError("NaturalSplineGrid: needs >= 3 knots")
        self.h = self.x[1:] - self.x[:-1]
        self.cp = np.zeros(n)
        self.den = np.ones(n)
        for i in range(1, n - 1):
            a, b = self.h[i - 1], 2.0 * (self.h[i - 1] + self.h[i])
            self.den[i] = b - a * self.cp[i - 1]
            self.cp[i] = self.h[i] / self.den[i]

    def moments(self, y):
        """Second derivatives; ``y`` has shape (..., n)."""
        y = np.asarray(y, dtype=float)
        n, h = self.x.size, self.h
        dp = np.zeros_like(y)
        m = np.zeros_like(y)
        for i in range(1, n - 1):
            r = 6.0 * ((y[..., i + 1] - y[..., i]) / h[i]
                       - (y[..., i] - y[..., i - 1]) / h[i - 1])
            dp[..., i] = (r - h[i - 1] * dp[..., i - 1]) / self.den[i]
        for i in range(n - 2, 0, -1):
            m[..., i] = dp[..., i] - self.cp[i] * m[..., i + 1]
        return m

    def eval(self, y, m, xq):
        """Spline of rows (y, m) at the points xq (clamped): (..., nq)."""
        x = np.clip(np.asarray(xq, dtype=float), self.x[0], self.x[-1])
        i = np.clip(np.searchsorted(self.x, x, side="right"), 1,
                    self.x.size - 1) - 1
        h = self.h[i]
        A = (self.x[i + 1] - x) / h
        B = (x - self.x[i]) / h
        h26 = h * h / 6.0
        return (A * y[..., i] + B * y[..., i + 1]
                + (A * A * A - A) * h26 * m[..., i]
                + (B * B * B - B) * h26 * m[..., i + 1])


class SelZFactored:
    """Factorised S_ij reader (SelZRule): S_rich_stack (richness factor,
    smooth in (lnM, z)) x exact S_j(z) x seam indicator; zero outside the
    table. ``tabulate`` is the grid evaluator the GL weights use
    (natural-spline by default, as sel_interp = spline)."""

    def __init__(self, source, spline=True):
        self.lnm = source.array("sel_function", "lnm")
        self.z = source.array("sel_function", "z")
        rich = np.asarray(source.array("sel_function", "s_rich_stack"),
                          dtype=float)
        if rich.ndim != 3:
            rich = rich.reshape(-1, self.z.size, self.lnm.size)
        self.n_bins = rich.shape[0]
        self.zob_min = np.atleast_1d(source.array("sel_function", "zob_min"))
        self.zob_max = np.atleast_1d(source.array("sel_function", "zob_max"))
        self.sigma_z = np.atleast_1d(source.array("sel_function", "sigma_z"))
        self.one_plus_z = (source.has("sel_function", "sigma_z_one_plus_z")
                           and bool(int(source.scalar(
                               "sel_function", "sigma_z_one_plus_z"))))
        lo = (source.scalar("sel_function", "zt_excl_low")
              if source.has("sel_function", "zt_excl_low") else 0.0)
        hi = (source.scalar("sel_function", "zt_excl_high")
              if source.has("sel_function", "zt_excl_high") else -1.0)
        self.excl = (lo, hi) if hi > lo else None
        self.spline = bool(spline)
        self._rich = rich
        self._interps = [
            RegularGridInterpolator((self.z, self.lnm), rich[b],
                                    method="linear")
            for b in range(self.n_bins)]

    @staticmethod
    def available(source):
        return all(source.has("sel_function", k) for k in
                   ("s_rich_stack", "sigma_z", "zob_min", "zob_max"))

    def s_j(self, b, z):
        z = np.asarray(z, dtype=float)
        w = self.sigma_z[b] * ((1.0 + z) if self.one_plus_z else 1.0)
        out = (ndtr((self.zob_max[b] - z) / w)
               - ndtr((self.zob_min[b] - z) / w))
        if self.excl is not None:
            out = np.where((z > self.excl[0]) & (z < self.excl[1]), 0.0, out)
        return out

    def panels(self, zt_lo, zt_hi, step_width=5.0):
        return composite_z_panels(
            self.zob_min, self.zob_max, self.sigma_z, zt_lo=zt_lo,
            zt_hi=zt_hi, excl=self.excl, one_plus_z=self.one_plus_z,
            step_width=step_width)

    def __call__(self, b, lnM, z):
        lnM_b, z_b = np.broadcast_arrays(np.asarray(lnM, dtype=float),
                                         np.asarray(z, dtype=float))
        inside = ((lnM_b >= self.lnm[0]) & (lnM_b <= self.lnm[-1])
                  & (z_b >= self.z[0]) & (z_b <= self.z[-1]))
        q = np.stack([np.clip(z_b, self.z[0], self.z[-1]),
                      np.clip(lnM_b, self.lnm[0], self.lnm[-1])], axis=-1)
        out = self._interps[b](q) * self.s_j(b, z_b)
        return np.where(inside, out, 0.0)

    def tabulate(self, b, lnm_nodes, z_nodes):
        """S_ij on (lnm_nodes x z_nodes), shape (n_lnm, n_z): SelZRule::
        tabulate -- tensor natural spline of S_rich (lnM pass, then z
        pass) x exact S_j, or pointwise bilinear when spline=False."""
        lnm_nodes = np.asarray(lnm_nodes, dtype=float)
        z_nodes = np.asarray(z_nodes, dtype=float)
        if not self.spline:
            return self(b, lnm_nodes[:, None], z_nodes[None, :])
        sm, sz = NaturalSplineGrid(self.lnm), NaturalSplineGrid(self.z)
        plane = self._rich[b]                              # (n_zt, n_mt)
        T = sm.eval(plane, sm.moments(plane), lnm_nodes)   # (n_zt, nk)
        col = T.T                                          # (nk, n_zt)
        out = sz.eval(col, sz.moments(col), z_nodes)       # (nk, nq)
        sj = self.s_j(b, z_nodes)
        inside = ((lnm_nodes[:, None] >= self.lnm[0])
                  & (lnm_nodes[:, None] <= self.lnm[-1])
                  & (z_nodes[None, :] >= self.z[0])
                  & (z_nodes[None, :] <= self.z[-1]) & (sj[None, :] != 0.0))
        return np.where(inside, out * sj[None, :], 0.0)


def sel_on_grid(sel, b, lnm_nodes, z_nodes):
    """S_ij of bin b on (lnm_nodes x z_nodes): the factorised reader's
    tabulate (SelZRule::tabulate) or the S_stack table pointwise."""
    if hasattr(sel, "tabulate"):
        return sel.tabulate(b, lnm_nodes, z_nodes)
    return sel(b, np.asarray(lnm_nodes)[:, None], np.asarray(z_nodes)[None, :])


def sel_z_rule(source, *, zt_lo, zt_hi, n_z=64, z_rule="composite",
               n_z_panel=8, z_step_width=5.0, sel_interp="spline"):
    """(z_x, z_w, sel) of the true-z rule -- the SelZRule twin.

    composite (default) with the factorised sel_function outputs present:
    composite GL + SelZFactored. Otherwise the single n_z GL panel on the
    bilinear S_stack table (previous behaviour, identical nodes).
    """
    if z_rule not in ("composite", "global"):
        raise ValueError(f"z_rule must be composite or global, got {z_rule!r}")
    if sel_interp not in ("spline", "bilinear"):
        raise ValueError(f"sel_interp must be spline or bilinear, "
                         f"got {sel_interp!r}")
    if z_rule == "composite" and SelZFactored.available(source):
        sel = SelZFactored(source, spline=(sel_interp == "spline"))
        z_x, z_w = composite_gl_nodes(
            sel.panels(zt_lo, zt_hi, z_step_width), n_z_panel)
        return z_x, z_w, sel
    z_x, z_w = gl_nodes(zt_lo, zt_hi, n_z)
    return z_x, z_w, SelStack(source)


class Bilinear2D:
    """Generic clamped-bilinear replica of make_Interp2D over a section.

    C++ convention: Interp2D(xs, ys, zs) with the value table stored
    rows = y, cols = x; queries are clamped to the domain.

    cubic=True mirrors Interp2D(..., bicubic=true): a bicubic spline
    (needs >= 4 points per axis) instead of bilinear, removing the
    derivative kinks at every knot that stall quadrature convergence.
    """

    def __init__(self, source, section, xkey, ykey, valkey, nan_fill=None,
                 cubic=False):
        self._x = source.array(section, xkey)
        self._y = source.array(section, ykey)
        vals = np.asarray(source.array(section, valkey), dtype=float)
        if vals.shape != (self._y.size, self._x.size):
            vals = vals.reshape(self._y.size, self._x.size)
        if nan_fill is not None:
            vals = np.where(np.isfinite(vals), vals, float(nan_fill))
        self._cubic = bool(cubic)
        if self._cubic:
            self._interp = RectBivariateSpline(self._x, self._y, vals.T,
                                               kx=3, ky=3, s=0)
        else:
            self._interp = RegularGridInterpolator(
                (self._y, self._x), vals, method="linear",
                bounds_error=False, fill_value=None)

    def __call__(self, x, y):
        x = np.clip(np.asarray(x, dtype=float), self._x[0], self._x[-1])
        y = np.clip(np.asarray(y, dtype=float), self._y[0], self._y[-1])
        y_b, x_b = np.broadcast_arrays(y, x)
        if self._cubic:
            return self._interp.ev(x_b, y_b)
        return self._interp(np.stack([y_b, x_b], axis=-1))


class SigmaCritInv:
    """Source-averaged Sigma_crit^-1(z_lens), linear with edge clamping."""

    def __init__(self, source):
        self._z = source.array("average_sigma_crit_inv", "zlense")
        self._sci = source.array("average_sigma_crit_inv", "sci_average")

    def __call__(self, z):
        z = np.clip(np.asarray(z, dtype=float), self._z[0], self._z[-1])
        return np.interp(z, self._z, self._sci)


# ---------------------------------------------------------------------------
# Miscentering helpers (lensing_weights.hh mis_detail)
# ---------------------------------------------------------------------------

F_MIS_DEFAULT = 0.22
TAU_MIS_DEFAULT = 0.17
DEFAULT_LOB_CENTERS = (25.0, 37.5, 52.5, 130.0)


def R_lambda(lob):
    """R_lambda(lambda^ob) = (lambda^ob / 100)^0.2 [Mpc/h]."""
    return (np.asarray(lob, dtype=float) / 100.0) ** 0.2


def read_mis_param(source, key, default):
    try:
        return source.scalar("miscentering", key)
    except Exception:
        return default


def physical_density_flag(source):
    """haloModel/one_halo_physical_density (0/1; absent -> off)."""
    try:
        return bool(int(source.scalar("halomodel", "one_halo_physical_density")))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixed-GL z-marginalised mass weights (SelGLCore replica)
# ---------------------------------------------------------------------------

class MassZWeights:
    """Per-sample W_ij(lnM) on fixed GL nodes, plus arbitrary moments.

        W_ij(lnM) = int dz n(M,z) dV/dOmega/dz Omega(z) S_ij(lnM,z)
                    [ * Sigma_crit_inv(z), shear only ]

    Mirrors nosel_gl_detail::SelGLCore::build_weights: identical node
    placement (GL on [lnm_lo, lnm_hi]; true z by ``sel_z_rule``, the
    SelZRule twin) and identical term composition, vectorised over
    (bin, lnM node, z node).
    """

    def __init__(self, source, *, n_lnm=96, n_z=64,
                 zt_lo, zt_hi, lnm_lo, lnm_hi, include_sci=False,
                 z_amp_power=0.0, z_rule="composite", n_z_panel=8,
                 z_step_width=5.0, sel_interp="spline"):
        """z_amp_power: fold (1+z)^p into the z weight -- the exact
        amplitude half of the physical-density identity
        DSigma_phys(R|z) = (1+z)^2 DSigma_com(R (1+z)). Shear consumers
        pass 2 when haloModel/one_halo_physical_density is on; number
        counts never pass it. Mirrors SelGLCore::build_weights."""
        self.lnm_x, self.lnm_w = gl_nodes(lnm_lo, lnm_hi, n_lnm)
        self.z_x, self.z_w, sel = sel_z_rule(
            source, zt_lo=zt_lo, zt_hi=zt_hi, n_z=n_z, z_rule=z_rule,
            n_z_panel=n_z_panel, z_step_width=z_step_width,
            sel_interp=sel_interp)

        hmf = HMF(source)
        dv = DVDoDz(source)
        self.n_bins = sel.n_bins

        zfac = self.z_w * dv(self.z_x) * omega_z_des(self.z_x)
        if include_sci:
            zfac = zfac * SigmaCritInv(source)(self.z_x)
        if z_amp_power != 0.0:
            zfac = zfac * (1.0 + self.z_x) ** z_amp_power

        lnm_grid = self.lnm_x[:, None]
        z_grid = self.z_x[None, :]
        hmf_kq = hmf(lnm_grid, z_grid)                      # (n_lnm, n_z)
        self.W = np.empty((self.n_bins, self.lnm_x.size))
        Wz = np.empty_like(self.W)
        for b in range(self.n_bins):
            s_kq = sel_on_grid(sel, b, self.lnm_x, self.z_x)  # (n_lnm, n_z)
            t = hmf_kq * s_kq
            self.W[b] = t @ zfac
            Wz[b] = t @ (zfac * self.z_x)
        # Selection-weighted mean redshift per bin (the query point for
        # the physical-density radius rescale in z-contracted evaluators).
        n0 = self.W @ self.lnm_w
        with np.errstate(divide="ignore", invalid="ignore"):
            self.z_eff = np.where(n0 != 0.0, (Wz @ self.lnm_w) / n0,
                                  0.5 * (zt_lo + zt_hi))

    def norm(self):
        """int dlnM W_ij for every bin — the NumCountsSel observable."""
        return self.W @ self.lnm_w

    def moments_of(self, y_of_lnm, ell_max=3):
        """Plain central moments of y(lnM) under each bin's weight.

        Returns (norm, ybar, mu) with mu[b, ell] for 0 <= ell <= ell_max;
        mu[:, 0] == 1 and mu[:, 1] == 0 by construction.
        """
        y = np.asarray(y_of_lnm(self.lnm_x), dtype=float)
        wW = self.W * self.lnm_w[None, :]
        norm = wW.sum(axis=1)
        ybar = (wW @ y) / norm
        d = y[None, :] - ybar[:, None]
        mu = np.empty((self.n_bins, ell_max + 1))
        for ell in range(ell_max + 1):
            mu[:, ell] = np.sum(wW * d**ell, axis=1) / norm
        return norm, ybar, mu
