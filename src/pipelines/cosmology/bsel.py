"""Close the CPP selection-bias integrals on the exact wall.

The CPP modules write one ``P1/I1/J`` row for every
``(lambda_bin, zo_low, zo_high)`` wall bin.  This module uses that same row
ordering and writes one ``b_small/b_large`` pair per row.  Consumers recover
the angular dependence analytically with the shared sigmoid; there is no
theta table and no interpolation between redshift bins.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time

import numpy as np
from cosmosis.datablock import option_section

_PIPELINES_DIR = str(Path(__file__).resolve().parents[1])
if _PIPELINES_DIR not in sys.path:
    sys.path.insert(0, _PIPELINES_DIR)

from cosmology.prj_params import PrjParams
from shared import datablock_models as dm


@dataclass
class IntegratorGLBSel:
    """Numerical engine that turns ``P1/I1/J`` into ``b_small/b_large``.

    The integrator processes one exact C++ wall row at a time. For each row it
    performs the following readable sequence:

    1. use the shared HMF and halo-bias interpolators on a common mass GL rule;
    2. calculate the HOD-weighted effective halo bias ``b_eff``;
    3. integrate over true richness ``ltr`` with Gauss-Legendre nodes;
    4. evaluate the small- and large-scale bias values at every ``ltr`` node;
    5. average those values with the HOD prior and the EMG projection kernel.

    The class does not create a theta grid. The theta dependence factorizes
    exactly in the C++ operators, so the only producer outputs are the two
    scalars ``b_small`` and ``b_large`` for each row.

    Configuration fields
    ---------------------
    ``ltr_lo`` / ``ltr_hi_factor`` / ``ltr_hi_fixed``
        True richness integration limits. By default the upper limit is
        ``ltr_hi_factor * lob``; a positive ``ltr_hi_fixed`` overrides it.
    ``n_ltr``
        Number of true richness Gauss-Legendre nodes per wall row.
    ``min_mass4integral`` / ``max_log10_mass`` / ``n_mass``
        Bounds and number of Gauss-Legendre nodes for the shared ``ln(M)``
        integration rule.
    ``verbose``
        Print one timing line after the wall has been processed.
    """

    ltr_lo: float
    ltr_hi_factor: float
    ltr_hi_fixed: float
    n_ltr: int
    min_mass4integral: float
    max_log10_mass: float
    n_mass: int
    verbose: bool = False
    mass: np.ndarray = field(init=False, repr=False)
    mass_weights: np.ndarray = field(init=False, repr=False)
    _ltr_nodes: np.ndarray = field(init=False, repr=False)
    _ltr_weights: np.ndarray = field(init=False, repr=False)
    _projection_parameters: PrjParams | None = field(
        init=False, default=None, repr=False)

    def __post_init__(self):
        """Build quadrature rules that do not depend on a datablock sample."""
        # The mass rule is fixed by [bsel] options, so construct it once in
        # setup rather than rebuilding the same nodes for every MCMC sample.
        ln_mass, self.mass_weights = dm.gl_nodes(
            np.log(self.min_mass4integral),
            self.max_log10_mass * np.log(10.0),
            self.n_mass,
        )
        self.mass = np.exp(ln_mass)

        # The ltr interval changes with lob, but the canonical [-1, 1] nodes
        # and weights do not. Cache the latter and map them per wall row.
        self._ltr_nodes, self._ltr_weights = np.polynomial.legendre.leggauss(
            self.n_ltr)

    @classmethod
    def from_options(cls, options):
        """Build the integrator from the CosmoSIS ``[bsel]`` options."""
        return cls(
            ltr_lo=options.get_double(option_section, "ltr_lo", default=1.0),
            ltr_hi_factor=options.get_double(
                option_section, "ltr_hi_factor", default=3.0),
            ltr_hi_fixed=options.get_double(
                option_section, "ltr_hi", default=0.0),
            n_ltr=options.get_int(option_section, "n_ltr", default=128),
            min_mass4integral=options.get_double(
                option_section, "min_mass4integral", default=1.0e13),
            max_log10_mass=options.get_double(
                option_section, "ln_M_max_log10", default=15.5),
            n_mass=options.get_int(option_section, "n_m_beff", default=100),
            verbose=options.get_bool(option_section, "verbose", default=False),
        )

    def get_projection_parameters(self, source):
        """Return the fixed projection calibration, loading it only once.

        CosmoSIS ``setup`` has no datablock, so a custom published
        ``plob_ltr_params`` table can only be discovered on the first
        ``execute`` call. The table is frozen for the run and is then reused
        for every subsequent MCMC sample.
        """
        if self._projection_parameters is None:
            self._projection_parameters = (
                PrjParams.from_source_or_default(source)
            )
        return self._projection_parameters

    def make_ltr_quadrature(self, lob):
        """Return true richness nodes and weights for one ``lob`` bin.

        The standard Legendre rule lives on ``[-1, 1]``. This method maps it
        to ``[ltr_lo, ltr_hi]`` and includes the Jacobian in the returned
        weights:

        ``ltr = midpoint + half_width * legendre_node``
        ``weight = half_width * legendre_weight``.
        """
        upper_ltr = (
            self.ltr_hi_fixed
            if self.ltr_hi_fixed > 0.0
            else self.ltr_hi_factor * float(lob)
        )
        if upper_ltr <= self.ltr_lo:
            raise ValueError(
                f"invalid true richness range [{self.ltr_lo}, {upper_ltr}]"
            )

        # The cached nodes x_i and weights w_i live on [-1, 1]. Mapping that
        # rule to [ltr_lo, upper_ltr] gives
        #
        #   ltr_i = midpoint + half_width * x_i
        #   w_i   = half_width * w_i^GL.
        #
        # The second equation is the Jacobian of the interval mapping.
        half_width = 0.5 * (upper_ltr - self.ltr_lo)
        midpoint = 0.5 * (upper_ltr + self.ltr_lo)
        return (
            midpoint + half_width * self._ltr_nodes,
            half_width * self._ltr_weights,
        )

    @staticmethod
    def evaluate_b_eff(lob, zob, phod: dm.PHOD, mass, number_density,
                       halo_bias, mass_weights):
        """Calculate the HOD-weighted effective halo bias for one wall row.

        The integral is

        ``b_eff = integral[dlnM * dn/dM * P_HOD(lob|M,z) * M * b(M,z)]``
        ``        / integral[dlnM * dn/dM * P_HOD(lob|M,z) * M]``.

        ``number_density`` is expected to be ``dn/dM`` when it enters this
        method. ``mass_weights`` are the Gauss-Legendre weights for ``dlnM``.
        The explicit ``M`` factor is therefore part of the weight.
        """
        ln_mass = np.log(mass)
        # PHOD returns P_HOD(lob | M, z). Since number_density is dn/dM and
        # the integral is dlnM, M supplies the explicit mass Jacobian:
        #
        #   b_eff(lob,z) =
        #     sum_i w_i * (dn/dM)_i * P_HOD_i * M_i * b_i
        #     ------------------------------------------------
        #     sum_i w_i * (dn/dM)_i * P_HOD_i * M_i,
        #
        # where w_i are the GL weights for dlnM.
        probability = phod(float(lob), ln_mass, float(zob))
        weight = number_density * probability * mass
        denominator = np.sum(mass_weights * weight)
        if denominator <= 0.0:
            return 0.0
        return float(np.sum(mass_weights * weight * halo_bias) / denominator)

    @staticmethod
    def evaluate_ltr_prior(ltr, zob, lob, phod: dm.PHOD, mass,
                           number_density, mass_weights,
                           projection_parameters: PrjParams):
        """Calculate the true richness marginalization weight.

        For each true richness node this returns

        ``weight(ltr) = P(lob | ltr, zob)``
        ``             * integral[dlnM * dn/dM * P_HOD(ltr|M,z) * M]``.

        The first factor is the canonical Gaussian/EMG projection kernel. The
        second factor is the mass-integrated HOD prior. The caller multiplies
        this result by the GL weight in ``ltr``. ``mass_weights`` are the
        separate Gauss-Legendre weights for the ``ln(M)`` integral.
        """
        ln_mass = np.log(mass)
        # Broadcast the HOD evaluation over every ltr node and every mass
        # point. This is the main vectorized part of the marginalization.
        probability = phod(ltr[:, None], ln_mass[None, :], float(zob))
        # The mass-integrated HOD prior is
        #
        #   prior(ltr,z) =
        #     sum_j w_j * (dn/dM)_j * P_HOD(ltr|M_j,z) * M_j.
        prior = np.sum(
            mass_weights[None, :]
            * number_density[None, :]
            * probability
            * mass[None, :],
            axis=1,
        )

        # PrjParams owns the analytical P(lob | ltr, zob) evaluation and its
        # z-interpolated EMG coefficients. bsel only consumes the probability.
        projection_probability = projection_parameters.p_lob_given_ltr(
            lob, ltr, zob)
        return prior * projection_probability

    @staticmethod
    def evaluate_b_large(lob, ltr, p1, i1, j, b_eff):
        """Evaluate ``b_large(ltr)`` for one wall row.

        The large-scale closure is

        ``I2 = I1 + J``
        ``Delta_RND = P1 + b_eff * I2``
        ``b_large(ltr) = b_eff * [1 + 0.13 * ((lob-ltr)/Delta_RND - 1)]``.

        ``ltr`` is an array of true richness GL nodes, so this method is
        vectorized over the true-richness integration.
        """
        # J is supplied directly by C++ as I2 - I1. Reconstruct I2 here only
        # for the analytical closure; do not form J by subtracting I2 - I1.
        i2 = i1 + j
        delta_rnd = p1 + b_eff * i2
        projection_shift = (lob - ltr) / delta_rnd - 1.0
        return b_eff * (1.0 + 0.13 * projection_shift)

    @staticmethod
    def evaluate_b_small(lob, ltr, p1, i1, j, b_large):
        """Evaluate ``b_small(ltr)`` for one wall row.

        The small-scale closure is

        ``b_small(ltr) = [(lob-ltr) - P1 - b_large(ltr)*I1] / J``.

        A vanishing ``J`` is a genuine singularity of the closure. Such
        nodes are returned as ``NaN`` so the row-level integrator can leave
        the output at its safe initialized value rather than masking it.
        """
        i2 = i1 + j
        denominator_scale = np.abs(i1) + np.abs(i2)
        valid_j = np.abs(j) > 1.0e-12 * denominator_scale
        return np.where(
            valid_j,
            ((lob - ltr) - p1 - b_large * i1) / j,
            np.nan,
        )

    def integrate_one_wall_row(
        self,
        lob,
        zob,
        p1,
        i1,
        j,
        phod: dm.PHOD,
        hmf: dm.HMF,
        halo_bias: dm.Bilinear2D,
        projection_parameters: PrjParams,
    ):
        """Integrate one exact ``(lob, zob)`` wall row.

        The C++ operators provide ``P1``, ``I1``, and ``J`` for this row. This
        method performs the complete mass and true-richness calculation and
        returns only the row's ``(b_small, b_large)`` pair. It deliberately
        knows nothing about wall-vector allocation or output writing.

        The shared ``ln(M)`` Gauss-Legendre rule was built once in
        ``__post_init__``. ``hmf`` and ``halo_bias`` are the datablock models
        for the current sample; they own the production interpolation and
        axis conventions.
        """
        # The mass nodes and weights are configuration-only state. Reusing
        # them here avoids rebuilding the same quadrature rule for every wall
        # row and every MCMC sample.
        mass = self.mass
        mass_weights = self.mass_weights
        ln_mass = np.log(mass)

        # HMF returns dn/dlnM. The HOD integrals below use
        #
        #   dn/dM = (dn/dlnM) / M,
        #
        # and then include M explicitly in the dlnM weight. The shared HMF
        # model applies the same mass-axis shift and nuisance factors used by
        # the rest of the fixed-GL Python pipeline.
        dndln_mass = hmf(ln_mass, zob)
        number_density = dndln_mass / mass
        halo_bias_values = halo_bias(ln_mass, zob)

        # b_eff is the halo bias averaged with the HOD probability for this
        # observed richness bin. It is independent of ltr and is therefore
        # computed once before the true-richness quadrature.
        effective_bias = self.evaluate_b_eff(
            lob,
            zob,
            phod,
            mass,
            number_density,
            halo_bias_values,
            mass_weights,
        )
        ltr, ltr_weights = self.make_ltr_quadrature(lob)

        # Evaluate the two analytical closure branches independently. Both
        # methods operate on the complete ltr GL vector without a Python loop.
        b_large_ltr = self.evaluate_b_large(lob, ltr, p1, i1, j, effective_bias)
        b_small_ltr = self.evaluate_b_small(lob, ltr, p1, i1, j, b_large_ltr)

        # Marginalize the two ltr-dependent bias values. For each GL node,
        #
        #   W_i = w_i^GL * P(lob|ltr_i,zob) * prior(ltr_i,zob),
        #
        #   b_small = sum_i W_i*b_small(ltr_i) / sum_i W_i,
        #   b_large = sum_i W_i*b_large(ltr_i) / sum_i W_i.
        ltr_prior = self.evaluate_ltr_prior(
            ltr,
            zob,
            lob,
            phod,
            mass,
            number_density,
            mass_weights,
            projection_parameters,
        )
        marginalization_weight = ltr_weights * ltr_prior
        normalization = np.sum(marginalization_weight)
        if normalization <= 0.0 or not np.all(np.isfinite(b_small_ltr)):
            return 0.0, 0.0

        return (
            float(np.sum(marginalization_weight * b_small_ltr)
                  / normalization),
            float(np.sum(marginalization_weight * b_large_ltr)
                  / normalization),
        )

    def integrate_b_small_large(
        self,
        wall,
        phod: dm.PHOD,
        hmf: dm.HMF,
        halo_bias: dm.Bilinear2D,
        projection_parameters: PrjParams,
    ):
        """Loop over the exact wall and write one pair per row.

        This method only coordinates the wall datavector. The numerical work
        for an individual ``(lob, zob, P1, I1, J)`` row lives in
        :meth:`integrate_one_wall_row`.
        """
        row_count = wall.lob.size
        b_small = np.zeros(row_count, dtype=float)
        b_large = np.zeros(row_count, dtype=float)

        for wall_row in range(row_count):
            # Preserve the C++ row ordering exactly while delegating the
            # calculation for each row to the focused integration method.
            b_small[wall_row], b_large[wall_row] = self.integrate_one_wall_row(
                wall.lob[wall_row],
                wall.zob[wall_row],
                wall.p1[wall_row],
                wall.i1[wall_row],
                wall.j[wall_row],
                phod,
                hmf,
                halo_bias,
                projection_parameters,
            )

        return dm.BSelOutputVector(
            wall.lambda_bin, wall.zo_low, wall.zo_high, wall.zob, wall.lob,
            b_small, b_large,
        ).validate()


def setup(options):
    return IntegratorGLBSel.from_options(options)


def execute(block, config):
    started = time.perf_counter()
    source = dm.DataBlockSource(block)

    # Lambda-bin edges are produced by sel_function. Reading them here keeps
    # the Python wall geometry identical to the C++ selection geometry.
    lambda_edges = source.array("sel_function", "lambda_edges")
    # Keep the continuous HOD as a model object. It owns the shifted-Poisson
    # and gamma-function evaluation; no HOD parameters are passed as a dict.
    hod_model = dm.PHOD.from_source(source)
    wall = dm.BSelWallVector.from_source(source, lambda_edges)

    # Reuse the shared HMF and bilinear halo-bias models. They implement the
    # same interpolation, clamping, mass-axis, and nuisance conventions used
    # by the other fixed-GL Python consumers.
    hmf = dm.HMF(source)
    halo_bias = dm.Bilinear2D(source, "halomodel", "lnm", "z", "bias")

    # Use the complete datablock calibration when published; otherwise use
    # the complete frozen default table. PrjParams owns this fallback policy.
    projection_parameters = config.get_projection_parameters(source)

    # The integrator returns one b_small/b_large pair per C++ wall row. The
    # datavector owns the output schema and writes it back to the datablock.
    output = config.integrate_b_small_large(
        wall, hod_model, hmf, halo_bias, projection_parameters)
    output.write_to_datablock(block)

    if config.verbose:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        print(f"[bsel.py] evaluated {wall.lob.size} exact wall rows "
              f"in {elapsed_ms:.1f} ms", flush=True)
    return 0


def cleanup(config):
    return 0
