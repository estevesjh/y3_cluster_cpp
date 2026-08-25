#!/usr/bin/env python3
"""Cross-backend / cross-strategy consistency for one-halo miscentred shear.

Verifies that every implemented (strategy, backend) cell agrees with
every other cell, per the matrix in docs/source/pipeline_organization.md:

    full_ltmz     -> Python, C++, CUDA        (explicit-selection references)
    fast_mass     -> Python, C++ (bitwise = production Shear1hMisSel.so)
    radial_series -> Python, C++ (offline U_ell tables + moments)
    max model     -> Python, C++, CUDA        (traditional 1h+2h; ΔΣ_hh has
                                               an open debugging flag, see
                                               docs/known_issues/dsigma_hh_debug_flag.md)

The reference is the C++ full_ltmz backend (Shear1h3d.so, adaptive
Cuhre eps_rel=1e-4). Its values, the CUDA full_ltmz backend's
(Shear1h3dGpu.so, PAGANI), the C++ radial_series backend's
(Shear1hRadialSeries.so), and the C++/CUDA max-model backends'
(Shear1h2hMax.so / Shear1h2hMaxGpu.so) are hard-coded below: reproducing
them needs a full CosmoSIS pipeline run of the compiled .so modules
(the max model additionally needs `compute_lensing_2h = T`, which the
repo's checked-in fiducial dump was NOT generated with -- see
docs/known_issues/dsigma_hh_debug_flag.md), and CUDA needs a GPU.

Provenance: `shear_gpu_smoke.ini`/`shear_extra_backends.ini` (2026-08-12
job 56790321 companion runs, and a 2026-08-14 companion run for the
fast_mass/radial_series/max-model C++/CUDA backends), same
`mock_mcmc_widePlanck_values.ini` fiducial point and pinned 12-bin x
10-radius wall used everywhere else in this repo's tests. To
regenerate: append the relevant module(s) to
cosmosis-models/real_pipeline_extract.ini (same bin wall; for the max model,
also set halo_model's `compute_lensing_2h = T`) and read back
`shear1h3d(gpu)/vals`, `shear1h_radial_series/vals`,
`shear1h2h_max(_gpu)/vals`.

The Python full_ltmz, fast_mass, and radial_series backends, and
production's own saved output, are all computed/read LIVE from this
repo's checked-in fiducial dump (cosmosis-models/real_pipeline_extract_output).

Note on radial_series vs full_ltmz (FINDING, real -- see
docs/known_issues/radial_series_vs_full_ltmz_defect.md): `nfw_profile_family.py`
hardcodes `CONC = 4.0` as a module-level constant, used for BOTH the
profile shape function and its amplitude normalization (r_s(M), A0(y)),
rather than reading the per-sample Child18 concentration
full_ltmz/fast_mass/production use. This is not just a shape/curvature
difference: raw ΔΣ values disagree by 56-86% (dominant, growing with
richness bin, i.e. with mass -- consistent with real concentration
decreasing with mass while c=4 stays fixed), on top of a further ~10%
shape/curvature residual once that amplitude offset is normalized out
(the number `validate_radial_series.py`'s own "check 4" already
reports, "the disclosed centred-profile convention gap"). Kept at the
project's standard tolerance rather than loosened or reported-only, per
policy: a failing test here is doing its job (see
test_cpp_radial_series_matches_cpp_full_ltmz below). The
C++-vs-Python identity check (test_python_radial_series_matches_
cpp_radial_series) stays separately, since it isolates the language/
interpolation-implementation question from this modeling-approximation
one.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))
_SHEAR1H2H = REPO / "src" / "pipelines" / "des_y3" / "shear_1h2h"
sys.path.insert(0, str(_SHEAR1H2H / "python" / "0d"))
sys.path.insert(0, str(_SHEAR1H2H / "python" / "0d"))
sys.path.insert(0, str(_SHEAR1H2H / "python" / "0d"))
sys.path.insert(0, str(_SHEAR1H2H / "python" / "0d"))

from shared import datablock_models as dm  # noqa: E402
from shared import lensing_profiles as lp  # noqa: E402
from systematics.selection_richness.python import sel_kernels  # noqa: E402
from shear1h_explicit_gl import compute_shear as compute_shear_full_ltmz  # noqa: E402
from shear1h_gl import compute_shear as compute_shear_fast_mass  # noqa: E402
import nfw_profile_family as pf  # noqa: E402
from shear1h_radial_series import RadialSeriesTable, evaluate_series  # noqa: E402

DUMP_DIR = REPO / "cosmosis-models" / "real_pipeline_extract_output"
HAS_DUMP = (DUMP_DIR / "matter_power_lin" / "p_k.txt").is_file()
_SKIP_MSG = (f"requires a real-pipeline dump at {DUMP_DIR} -- run "
            "`cosmosis cosmosis-models/real_pipeline_extract.ini` first")

R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                  1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
BINS = dict(
    lam_min=np.array([20., 30., 45., 60.] * 3),
    lam_max=np.array([30., 45., 60., 200.] * 3),
    zob_min=np.array([0.20] * 4 + [0.35] * 4 + [0.50] * 4),
    zob_max=np.array([0.35] * 4 + [0.50] * 4 + [0.65] * 4),
    sigma_z=np.full(12, 0.03),
)
ZT_LOW, ZT_HIGH = 0.05, 0.80
LNM_LOW, LNM_HIGH = 29.9336, 36.7300

# C++ full_ltmz (Shear1h3d.so) -- THE reference. See module
# docstring for provenance.
CPP_FULL_LTMZ = np.array([
    109728.63274876587, 93049.48506585791, 75636.26207846176, 58403.52821044179,
    42469.85403208658, 29088.145314104888, 18907.93116302601, 11803.39987252568,
    7125.237989409471, 4187.673496800235, 52468.524003604514, 45869.81584483353,
    38526.59667227423, 30785.983151682987, 23166.92759008317, 16373.572728563195,
    10932.2641574797, 6977.196069675863, 4289.735274328181, 2559.618845205281,
    17466.988475644604, 15632.757441707352, 13479.224641603772, 11080.693758893323,
    8585.256683283422, 6237.571247213942, 4266.35974108226, 2778.297853273661,
    1737.0750002322013, 1051.0021863664758, 13704.60556043571, 12526.5700803612,
    11087.149235300132, 9413.25838552937, 7573.878138136094, 5720.9877077790925,
    4051.9646624446646, 2715.9961855375727, 1739.1482496182996, 1073.309150897924,
    177109.44015635885, 149627.95030135766, 121176.03656977568, 93239.19281647485,
    67582.5441171783, 46154.395005636165, 29926.200079740236, 18642.259515969516,
    11233.835177315026, 6592.805331273523, 78105.33219242287, 68061.00932117167,
    56972.32345563049, 45374.486598246855, 34037.76453427126, 23987.574031580105,
    15975.202139494662, 10173.560275533318, 6243.512323093826, 3719.7622996461378,
    23722.066171654657, 21173.265383913014, 18202.56588139455, 14918.542000014711,
    11525.009004314305, 8350.516295090472, 5697.450990963002, 3702.310079308695,
    2310.5989051217634, 1395.8897003203128, 16027.383983789443, 14599.83717697898,
    12871.346643559724, 10881.778070016571, 8717.589227367695, 6557.093446954769,
    4625.666691923486, 3089.4660112047463, 1972.134473829814, 1213.873294796501,
    181504.44244796663, 152466.4089912783, 122778.45412028937, 93970.07934011379,
    67778.81971049927, 46085.924325467204, 29768.22448654382, 18484.637532794393,
    11109.13477045783, 6505.235146203046, 74543.40157502069, 64636.11578680574,
    53827.631244171345, 42653.86905501683, 31844.004466924685, 22343.386642529826,
    14822.462623897756, 9408.188927314795, 5757.701131644486, 3422.367680103516,
    20822.13834017465, 18509.025232930548, 15841.45263270001, 12924.535289494088,
    9940.382929079611, 7172.447829888165, 4875.250051629235, 3157.6745073204775,
    1965.2171964702145, 1184.4775229922816, 12349.700520480294, 11202.012847771082,
    9827.63361569311, 8265.109995659377, 6586.183983666125, 4928.368327756824,
    3459.8597834740954, 2300.8178845072316, 1463.1794278025386, 897.7266209016824,
])

# CUDA full_ltmz (Shear1h3dGpu.so, PAGANI).
CUDA_FULL_LTMZ = np.array([
    109729.56284061921, 93051.75518928096, 75638.38155615542, 58404.504161389734,
    42470.5710250148, 29088.60521033607, 18908.244294483356, 11803.60207342915,
    7125.360109918303, 4187.739963953105, 52468.84912691401, 45870.073969590565,
    38526.81859870375, 30785.93801473924, 23166.831387940583, 16373.47894252828,
    10932.349896633914, 6977.242330429006, 4289.764589727511, 2559.633363822753,
    17467.056531691665, 15632.843736899886, 13479.260344783077, 11080.696322440666,
    8585.258260598574, 6237.567281911035, 4266.358970889807, 2778.3029560004506,
    1737.0805845324421, 1051.0054251359388, 13704.979016235076, 12526.919926038929,
    11087.43425291467, 9413.506316848196, 7574.032003237256, 5721.1004091744835,
    4052.0291956021674, 2716.0374014046492, 1739.1740022916335, 1073.3236141842451,
    177105.99015949358, 149624.8035575659, 121173.24021722897, 93238.1075764932,
    67581.69505988012, 46153.88316601372, 29925.876853202317, 18642.084745650864,
    11233.757057088062, 6592.753342263139, 78104.88534468954, 68060.91903159283,
    56972.172450745355, 45374.4692298116, 34036.5800377164, 23986.712850215656,
    15974.63887354426, 10173.150779482927, 6243.25639960645, 3719.6975067016356,
    23721.992936163122, 21173.213049007027, 18202.478496851607, 14918.509817346001,
    11524.998506132202, 8350.568774099396, 5697.488423221311, 3702.314806687481,
    2310.6003438261946, 1395.8917684637727, 16027.755810106424, 14600.176226838244,
    12871.645050596751, 10882.026387895548, 8717.797212016916, 6557.229144623048,
    4625.758290130136, 3089.5251475648865, 1972.1709525740666, 1213.889217265928,
    181504.84328741755, 152466.53448221742, 122778.7666354857, 93970.05454134487,
    67778.77624393725, 46085.99406149496, 29768.351414168374, 18484.73407014158,
    11109.214837087227, 6505.26822700739, 74543.10284870176, 64635.8366583272,
    53827.189993044594, 42653.47136543446, 31843.64026879982, 22343.14927470194,
    14822.239593080867, 9408.144531620168, 5757.671785151373, 3422.385620248415,
    20822.047099981795, 18508.95437453866, 15841.368526125501, 12924.47984655705,
    9940.260088222563, 7172.392091648938, 4875.244231935678, 3157.6742559724553,
    1965.215280791274, 1184.4749513579568, 12350.171103391933, 11202.449268760658,
    9828.094001951855, 8265.561693433778, 6586.580030510737, 4928.610847495364,
    3460.0210069679065, 2300.9355981544522, 1463.246071173168, 897.7703686373897,
])

# C++ radial_series (Shear1hRadialSeries.so, ell_max=2 default).
CPP_RADIAL_SERIES = np.array([
    98730.42169358414, 85706.34946678484, 71302.24714856083, 56228.56520970219,
    41639.08024621691, 28947.73380537987, 19063.059225617544, 12021.513250421822,
    7322.815005202493, 4334.903423845499, 48372.96663543758, 42954.96983178918,
    36666.000153039626, 29748.02360723042, 22690.09619287764, 16215.94249311044,
    10935.033222474876, 7033.16695097371, 4353.5506947197255, 2612.34101350488,
    16504.720837003937, 14914.830138838171, 12993.897445139193, 10789.64137138133,
    8436.582412660422, 6176.3163218290565, 4253.609744690189, 2784.481219617833,
    1749.4701793758563, 1062.5992863964377, 13366.95386704789, 12267.403864590655,
    10905.430286193328, 9298.654464801271, 7510.8151781642055, 5690.091057554262,
    4040.877993580741, 2712.877884515725, 1739.7742989752091, 1074.892725923193,
    158996.6350907304, 137630.30818019615, 114163.95081069348, 89768.5001692467,
    66292.79732338799, 45969.91213564924, 30204.842359337377, 19011.521265162704,
    11562.172560002038, 6835.345005862895, 71817.3483180483, 63617.31797365546,
    54161.55021430457, 43825.94962824538, 33341.55611839577, 23770.21615589425,
    15994.171917935586, 10268.122019094364, 6346.24869808485, 3802.7237006813593,
    22354.794766031468, 20159.55573607043, 17522.7906414348, 14515.225441473714,
    11322.486860582787, 8269.81292648109, 5683.561311388189, 3713.9461142558102,
    2329.489323806536, 1413.150831095906, 15569.955544354223, 14250.523625759895,
    12628.370022342418, 10730.60680754896, 8636.435321681858, 6519.5534063191035,
    4614.273557159645, 3088.373131454979, 1975.3375858484578, 1217.6897983407891,
    162269.12760637054, 139854.94325437138, 115493.24535291507, 90415.88943726002,
    66492.8158636441, 45928.33964076884, 30077.21674139854, 18878.65621365811,
    11451.674444134493, 6756.591628704379, 68222.72916588932, 60208.631030326185,
    51055.37712287234, 41147.107229311354, 31180.92056196563, 22149.63562854272,
    14854.865605106941, 9509.094092210193, 5863.747388948489, 3506.8203703063596,
    19528.788255333606, 17555.97737050649, 15206.580473514943, 12550.761710903937,
    9754.391952545073, 7099.61710814619, 4863.625230022577, 3169.146832479224,
    1983.1697090633797, 1200.616219251731, 11930.692445073855, 10882.781538910423,
    9606.23448680467, 8127.910501666849, 6512.848787072882, 4895.048366286093,
    3450.1249274087254, 2300.474698168853, 1466.818692711358, 901.6859600952454,
])

# C++ max model (Shear1h2hMax.so). Requires compute_lensing_2h = T.
CPP_MAX_MODEL = np.array([
    109712.43068528689, 93034.86655675278, 75623.4492040774, 58393.12355381141,
    42461.96052384082, 29082.48457927022, 18904.166316868454, 11801.001291453327,
    7123.766262736592, 4187.446838605065, 52459.25029235579, 45860.81147270412,
    38518.26839680978, 30778.682263108156, 23161.03013094175, 16369.153310487867,
    10929.17853499355, 6975.146324405041, 4288.435463329837, 2558.828086136444,
    17463.956138316542, 15629.692220922117, 13476.24459385385, 11077.959410655121,
    8582.939433600874, 6235.748217497857, 4265.031084210242, 2777.3892761845673,
    1736.485386099238, 1050.6334461174417, 13707.326773727369, 12528.69756413135,
    11088.657789250174, 9414.215446809367, 7574.392390257829, 5721.191346089366,
    4051.990132737612, 2715.9451020744496, 1739.0779416309356, 1073.2462598336526,
    177078.6792058498, 149600.1431739446, 121152.05355452154, 93220.21553921321,
    67568.30288420372, 46144.34941452024, 29919.51460605116, 18638.030607157805,
    11231.233872250392, 6592.381564279313, 78095.30453693263, 68051.09999264409,
    56962.96966884831, 45366.30785643538, 34031.057624668676, 23982.49712549383,
    15971.62551170503, 10171.187466736992, 6242.0091041013175, 3718.850190547989,
    23721.222269210084, 21172.059194048466, 18201.118783895658, 14917.028823300921,
    11523.581322578799, 8349.31592468535, 5696.533679917137, 3701.658254961123,
    2310.164078796154, 1395.6144403247665, 16035.646207791891, 14606.996664135335,
    12877.295815238838, 10886.485618684475, 8721.112989522466, 6559.565899322946,
    4627.299006726878, 3090.4907423092664, 1972.7538267580928, 1214.236319241937,
    181306.1801637009, 152299.67749901605, 122644.28633577503, 93866.98876132947,
    67704.41383550814, 46035.241935046666, 29735.50204007882, 18464.29189297244,
    11096.9137089127, 6499.760860384868, 74480.05751702751, 64580.37554789032,
    53780.4397783235, 42615.87658142506, 31815.271340409057, 22322.9757224268,
    14808.791493584586, 9399.440773555345, 5752.31372478675, 3419.1711203303344,
    20809.983221988634, 18497.856155712776, 15831.537593635654, 12916.170717972087,
    9933.7422621194, 7167.514128879686, 4871.8110850206285, 3155.4044230904105,
    1963.7805780779358, 1183.599195919683, 12352.994590123579, 11204.711578861516,
    9829.72641509661, 8266.629544318448, 6587.223262003242, 4929.011898021327,
    3460.2303349932704, 2301.015953044335, 1463.2799277679462, 897.7764318066369,
])

# CUDA max model (Shear1h2hMaxGpu.so). Requires compute_lensing_2h = T.
CUDA_MAX_MODEL = np.array([
    109712.43068528747, 93034.86655675313, 75623.44920407781, 58393.1235538115,
    42461.96052384088, 29082.484579270287, 18904.166316868585, 11801.001291453349,
    7123.7662627366135, 4187.446838605101, 52459.250292355944, 45860.81147270414,
    38518.26839680991, 30778.682263108294, 23161.030130941832, 16369.153310487909,
    10929.178534993582, 6975.146324405068, 4288.43546332986, 2558.8280861364597,
    17463.956138316607, 15629.692220922148, 13476.244593853888, 11077.959410655134,
    8582.939433600928, 6235.748217497869, 4265.031084210254, 2777.3892761845755,
    1736.4853860992425, 1050.6334461174422, 13707.326773727404, 12528.697564131377,
    11088.657789250194, 9414.215446809376, 7574.392390257825, 5721.191346089359,
    4051.9901327376124, 2715.9451020744536, 1739.0779416309388, 1073.246259833657,
    177078.67920585044, 149600.14317394485, 121152.05355452167, 93220.21553921333,
    67568.30288420389, 46144.34941452027, 29919.514606051147, 18638.030607157878,
    11231.233872250401, 6592.381564279325, 78095.30453693273, 68051.09999264429,
    56962.9696688484, 45366.30785643547, 34031.05762466881, 23982.49712549388,
    15971.625511705084, 10171.187466737001, 6242.009104101317, 3718.850190547994,
    23721.222269210084, 21172.059194048536, 18201.11878389571, 14917.028823300892,
    11523.581322578813, 8349.315924685385, 5696.533679917154, 3701.658254961127,
    2310.16407879616, 1395.6144403247715, 16035.646207791895, 14606.996664135344,
    12877.29581523885, 10886.485618684486, 8721.1129895225, 6559.565899322959,
    4627.299006726875, 3090.4907423092723, 1972.7538267580908, 1214.236319241941,
    181306.180163701, 152299.6774990161, 122644.28633577471, 93866.9887613297,
    67704.41383550843, 46035.24193504687, 29735.502040078864, 18464.291892972466,
    11096.913708912714, 6499.7608603848885, 74480.05751702769, 64580.37554789044,
    53780.43977832369, 42615.876581425255, 31815.271340409123, 22322.97572242687,
    14808.791493584604, 9399.440773555361, 5752.313724786759, 3419.171120330339,
    20809.983221988667, 18497.856155712845, 15831.537593635672, 12916.170717972127,
    9933.742262119442, 7167.514128879697, 4871.811085020641, 3155.4044230904133,
    1963.7805780779406, 1183.5991959196867, 12352.994590123608, 11204.71157886158,
    9829.726415096622, 8266.629544318484, 6587.223262003263, 4929.011898021356,
    3460.230334993274, 2301.0159530443457, 1463.279927767944, 897.776431806639,
])


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestShear1hCrossBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = dm.DumpSource(str(DUMP_DIR))
        cls.mor = sel_kernels.mor_from_source(cls.source)
        cls.plob = sel_kernels.plob_splines_default()
        cls.hmf = dm.HMF(cls.source)
        cls.dv = dm.DVDoDz(cls.source)
        cls.sci = dm.SigmaCritInv(cls.source)
        cls.profile = lp.MisMixtureProfile(
            cls.source, lob_centers=dm.DEFAULT_LOB_CENTERS,
            f_mis=dm.F_MIS_DEFAULT, tau_mis=dm.TAU_MIS_DEFAULT,
            omega_m=cls.source.scalar("cosmological_parameters", "omega_m"))

    def _fast_mass_python(self):
        weights = dm.MassZWeights(
            self.source, n_lnm=96, n_z=64, zt_lo=ZT_LOW, zt_hi=ZT_HIGH,
            lnm_lo=LNM_LOW, lnm_hi=LNM_HIGH, include_sci=True)
        return compute_shear_fast_mass(weights, self.profile, np.arange(12),
                                       R_PERP)

    def test_python_full_ltmz_matches_cpp_full_ltmz(self):
        # Two completely different quadrature strategies (fixed-GL vs
        # adaptive Cuhre) over the same physics -- measured 3.3e-4.
        vals = compute_shear_full_ltmz(
            BINS, self.mor, self.plob, self.hmf, self.dv, self.sci,
            self.profile, np.arange(12), R_PERP,
            zt_low=ZT_LOW, zt_high=ZT_HIGH, lnm_low=LNM_LOW, lnm_high=LNM_HIGH)
        np.testing.assert_allclose(vals, CPP_FULL_LTMZ, rtol=6e-4)

    def test_cuda_full_ltmz_matches_cpp_full_ltmz(self):
        # Two independent adaptive integrators (PAGANI vs Cuhre) over the
        # identical integrand -- measured 8.4e-5.
        np.testing.assert_allclose(CUDA_FULL_LTMZ, CPP_FULL_LTMZ, rtol=2e-4)

    def test_python_fast_mass_matches_cpp_full_ltmz(self):
        # fast_mass's S_ij tabulation vs the explicit triple integral --
        # measured 1.1e-3.
        np.testing.assert_allclose(self._fast_mass_python(), CPP_FULL_LTMZ,
                                   rtol=2e-3)

    def test_production_fast_mass_matches_cpp_full_ltmz(self):
        vals = self.source.array("shear1hmissel", "vals")
        np.testing.assert_allclose(vals, CPP_FULL_LTMZ, rtol=2e-3)

    def test_python_fast_mass_matches_production_to_near_machine_precision(self):
        # Python fast_mass is a direct re-expression of the same
        # algorithm Shear1hMisSel.so implements (SelGLCore) -- measured
        # ~1e-7 (separate pipeline executions of the same deterministic
        # algorithm; not bitwise across two different processes, but
        # far tighter than any physics-level tolerance).
        prod = self.source.array("shear1hmissel", "vals")
        np.testing.assert_allclose(self._fast_mass_python(), prod, rtol=1e-5)

    def test_python_radial_series_matches_cpp_radial_series(self):
        # C++ vs Python radial_series, identical U_ell tables and
        # profile family -- an interpolation-scheme-only difference,
        # measured 1.6e-4 (docs/source/pipeline_organization.md's
        # matrix: "radial_series / C++ ... 3.7e-3 + 1.6e-4 interp-scheme
        # difference vs Python"). Deliberately NOT compared to
        # full_ltmz/production here -- see module docstring.
        table = RadialSeriesTable()
        weights = dm.MassZWeights(
            self.source, n_lnm=96, n_z=64, zt_lo=ZT_LOW, zt_hi=ZT_HIGH,
            lnm_lo=LNM_LOW, lnm_hi=LNM_HIGH, include_sci=True)
        rho_ref = self.source.scalar("halomodel", "rho_m_ref")
        norm, ybar, mu = weights.moments_of(
            lambda lnm: pf.y_of_lnM(lnm, rho_ref), ell_max=3)
        f_mis, tau_mis = dm.F_MIS_DEFAULT, dm.TAU_MIS_DEFAULT
        lob = np.asarray(dm.DEFAULT_LOB_CENTERS)

        vals = np.empty((12, R_PERP.size))
        for b in range(12):
            r_mis = tau_mis * float(dm.R_lambda(lob[b % lob.size]))
            vals[b] = evaluate_series(table, R_PERP, r_mis, norm[b], ybar[b],
                                      mu[b], f_mis=f_mis, rho_ref=rho_ref,
                                      ell_max=2)
        np.testing.assert_allclose(vals.ravel(), CPP_RADIAL_SERIES, rtol=3e-4)

    def test_cpp_radial_series_matches_cpp_full_ltmz(self):
        # FINDING (real, at the project's standard 1e-3 tolerance -- see
        # docs/known_issues/radial_series_vs_full_ltmz_defect.md). The
        # historical 56-86% disagreement was dominated by the density
        # convention mismatch (rho_crit/200c family vs the production
        # rho_m0/200m) and is RESOLVED by the unified
        # haloModel/rho_m_ref convention (2026-08-24): under the
        # regenerated pins (z_halo = 0 tables, matching the dump) the
        # residual envelope is -10.6% .. +3.9%
        # (sign flipping with radius within each bin). What remains is
        # the documented CONC = 4.0 defect: nfw_profile_family.py
        # hardcodes the concentration for both the profile shape and
        # its amplitude, instead of the per-sample Child18 relation the
        # 3d reference/production interpolate from haloModel/dSigma_nfw
        # -- a shape+amplitude residual that grows with mass. Kept at
        # the project's default tolerance rather than loosened, per
        # policy -- a failing test here is doing its job (issue #5).
        np.testing.assert_allclose(CPP_RADIAL_SERIES, CPP_FULL_LTMZ, rtol=1e-3)

    def test_cuda_max_model_matches_cpp_max_model(self):
        # Both backends' inner NFW-mixture contraction is bit-identical
        # in structure; measured ~7e-15 (machine precision).
        np.testing.assert_allclose(CUDA_MAX_MODEL, CPP_MAX_MODEL, rtol=1e-10)

    def test_max_model_is_never_less_than_1h_only(self):
        # Model invariant: Phi_max(R, lnM, z) = max(DSigma_cl, b*DSigma_hh)
        # >= DSigma_cl always, so the selection-weighted integral of the
        # max model can only be >= the pure-1h fast_mass integral, for
        # every (bin, R) -- true regardless of the ΔΣ_hh open defect
        # (docs/known_issues/dsigma_hh_debug_flag.md).
        #
        # Pad is 1e-6, NOT 1e-9 (issue #23): the pinned arrays and the
        # dump come from different executions, and cross-run float
        # reproducibility is ~1e-7 (measured in test_python_fast_mass_
        # matches_production_to_near_machine_precision below). In bins
        # where the 2h term never wins, the max equals the 1h branch
        # exactly and the ratio is 1.0 up to that noise, so a tighter pad
        # flips sign on reproducibility, not physics. 1e-6 still catches
        # any real sign/weighting bug (those show up at percent level;
        # the 2h term moves values at 1e-3+ where it wins). Same-run
        # regeneration: cosmosis-models/real_pipeline_extract_max2h.ini.
        fast_cpp_proxy = self.source.array("shear1hmissel", "vals")
        self.assertTrue(np.all(CPP_MAX_MODEL >= fast_cpp_proxy * (1.0 - 1e-6)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
