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
docs/figs/real_pipeline_extract.ini (same bin wall; for the max model,
also set halo_model's `compute_lensing_2h = T`) and read back
`shear1h3d(gpu)/vals`, `shear1h_radial_series/vals`,
`shear1h2h_max(_gpu)/vals`.

The Python full_ltmz, fast_mass, and radial_series backends, and
production's own saved output, are all computed/read LIVE from this
repo's checked-in fiducial dump (docs/figs/real_pipeline_extract_output).

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

DUMP_DIR = REPO / "docs" / "figs" / "real_pipeline_extract_output"
HAS_DUMP = (DUMP_DIR / "matter_power_lin" / "p_k.txt").is_file()
_SKIP_MSG = (f"requires a real-pipeline dump at {DUMP_DIR} -- run "
            "`cosmosis docs/figs/real_pipeline_extract.ini` first")

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
    106973.00873038164, 89182.7622178334, 70884.60892666598, 53408.59567669171,
    38006.578847691344, 25627.29264796322, 16486.133085834495, 10219.660160896618,
    6137.158783216153, 3592.6035518377544, 51274.96700233601, 44134.913986662155,
    36298.681764044675, 28321.75866504428, 20847.17083881875, 14490.917233582892,
    9567.945472879, 6061.467092390626, 3706.836978479537, 2202.928348467392,
    17099.54792373366, 15084.27834786115, 12750.436257266283, 10241.159418543513,
    7760.012642555974, 5540.719435241071, 3745.1514548437067, 2420.036475375677,
    1504.7981476682126, 906.7787832837323, 13479.55403048565, 12173.881305902622,
    10588.129829703834, 8793.19074700192, 6912.331294191013, 5118.5550139768675,
    3574.4890175123705, 2374.3852805343095, 1511.2655466794522, 928.7163075497699,
    172667.91474466887, 143406.55091146214, 113548.82919098195, 85244.40387138003,
    60460.582755635645, 40648.80465347448, 26083.972321585894, 16135.35785582051,
    9672.695176025789, 5654.0167606927735, 76331.5092793426, 65487.1635819857,
    53674.20590705209, 41735.523376944744, 30621.759822758217, 21223.317745473512,
    13977.395602208393, 8835.723765506635, 5393.56006083852, 3200.4675869882826,
    23224.27646120272, 20431.065569613413, 17217.96739905118, 13786.720569145706,
    10415.20541544351, 7415.877383172904, 5000.186349598829, 3224.1074999521693,
    2001.140174888689, 1204.0401204864413, 15765.32467651113, 14189.724864682581,
    12292.246895136748, 10164.118161769542, 7954.476260600033, 5864.862183219604,
    4079.227529651011, 2699.97210228301, 1713.1468464777633, 1049.98750400604,
    176961.11472136018, 146120.57311332852, 115027.17759153797, 85879.3704365353,
    60605.148861662514, 40566.17691929067, 25932.03937154085, 15990.251610381958,
    9560.176782526185, 5575.92293506095, 72856.16007829028, 62193.17682932892,
    50706.169446585525, 39222.68182565645, 28636.808980263646, 19759.626210232338,
    12962.787821648877, 8167.240551441487, 4971.578332295924, 2943.216155363737,
    20386.671156255776, 17860.99464728048, 14983.953358502218, 11941.910477677284,
    8980.500354944455, 6367.353230203406, 4276.973470056279, 2748.7688441974533,
    1701.359882085395, 1021.2843546830393, 12148.805686094658, 10888.200736227916,
    9385.6459597731, 7719.176087968008, 6008.077828523694, 4406.440750944536,
    3049.8779446309213, 2009.9107928184026, 1270.4978315397923, 776.1969190847334,
])

# CUDA full_ltmz (Shear1h3dGpu.so, PAGANI).
CUDA_FULL_LTMZ = np.array([
    106974.60871599588, 89184.1147310699, 70885.74060913107, 53409.469311380875,
    38007.20538406441, 25627.773711298505, 16486.422832803688, 10219.861693609255,
    6137.253695241021, 3592.657208156511, 51275.081160155, 44134.96064514124,
    36298.72537710132, 28321.719166554336, 20846.99513097083, 14490.831207257794,
    9567.9633880616, 6061.490609073303, 3706.849290407602, 2202.9378223580675,
    17099.578399343998, 15084.308237401927, 12750.453345679103, 10241.211965095741,
    7760.042615513545, 5540.7388229377275, 3745.160193415072, 2420.042555871645,
    1504.801485692084, 906.780366517468, 13479.728966242372, 12174.053576801534,
    10588.292942468704, 8793.327285055166, 6912.421176167303, 5118.607144598786,
    3574.5246869727944, 2374.4113767105737, 1511.2814899192806, 928.7261752231614,
    172666.1090517543, 143404.699268565, 113547.2165351232, 85243.2921813085,
    60459.78308762573, 40648.36051476134, 26083.770289186607, 16135.307775341324,
    9672.60372147646, 5653.971534446364, 76331.86548756257, 65487.35712578892,
    53674.06151683575, 41735.453777093586, 30621.63136172845, 21223.16423228646,
    13977.26324430913, 8835.725244098077, 5393.540246233363, 3200.4566507225472,
    23224.323351386673, 20431.093740819393, 17217.84933051761, 13786.63882074784,
    10415.143085918395, 7415.833615618378, 5000.168354301246, 3224.0963547656997,
    2001.1438108232446, 1204.0442474641748, 15765.65609662638, 14190.146181840879,
    12292.603650649286, 10164.419124241354, 7954.6885696061145, 5865.033581930039,
    4079.341214386519, 2700.0450617031015, 1713.1925348432037, 1050.0136119950982,
    176962.1461978616, 146121.91943383022, 115031.05117230726, 85881.80626331156,
    60607.498313335804, 40568.87825926908, 25933.849854465494, 15991.434830634422,
    9560.270744783487, 5575.966955667654, 72855.84400622454, 62192.920321195896,
    50706.00766467469, 39222.565255578535, 28636.655632113943, 19759.46843269308,
    12962.708111890985, 8167.227359535558, 4971.567027373856, 2943.202411265119,
    20387.122720686624, 17861.38797811597, 14984.224983428823, 11942.194099938992,
    8980.711691044075, 6367.493540248541, 4277.070668813594, 2748.836452767247,
    1701.3655490596504, 1021.2875395299617, 12149.781933878803, 10889.0931438849,
    9386.436621983356, 7719.602773776309, 6008.404217447584, 4406.639793117485,
    3050.016581518811, 2010.0058009795787, 1270.5548490042277, 776.2174318616361,
])

# C++ radial_series (Shear1hRadialSeries.so, ell_max=2 default).
CPP_RADIAL_SERIES = np.array([
    169395.31129056535, 135877.58060486315, 103731.1044369486, 75206.70420467714,
    51744.76716674016, 33897.95744591859, 21301.451765703096, 12936.72764027507,
    7639.346004375912, 4408.973230638092, 85649.02607977676, 70536.71676075064,
    55323.50456125513, 41195.88425297141, 29074.57367508011, 19485.836471411138,
    12480.806494267492, 7701.979120108262, 4609.252747683847, 2689.1357476690905,
    29945.288954351086, 25192.20306917884, 20206.91474551749, 15392.713754047774,
    11105.362537617422, 7593.733735100753, 4949.230790423497, 3098.593657145832,
    1876.5572466854096, 1105.8511572648235, 25018.053008247138, 21531.744652680143,
    17713.440457603163, 13867.307958375342, 10294.64752952637, 7236.771896432555,
    4831.9870307865585, 3086.9446584204775, 1901.0644888264037, 1135.7578708165313,
    271970.50250007154, 217502.5457553849, 165557.1151754739, 119701.0358860195,
    82152.70925458506, 53699.87140188183, 33682.01105518308, 20423.39763699199,
    12044.492642010413, 6943.9102259323745, 126830.74836780727, 104172.1263103329,
    81483.44156083342, 60518.28371902884, 42608.750186238, 28496.216470022173,
    18219.118660805027, 11224.864994131578, 6709.12761349057, 3910.2591734758435,
    40471.02636648363, 33966.7909368748, 27178.15243098606, 20652.593184188543,
    14866.464169926376, 10145.010395101819, 6599.398045060329, 4125.627079459543,
    2495.553499095971, 1468.9234454738823, 29058.554463726134, 24928.56611021722,
    20435.265127455565, 15939.773695952868, 11791.030041329937, 8261.050136988959,
    5499.361947628815, 3504.1488699401975, 2153.221959666395, 1284.0193416560767,
    276289.7605720848, 219950.49858970218, 166677.93796371578, 120004.40984814253,
    82052.44962880276, 53462.46979736836, 33433.965923073025, 20224.408635598902,
    11905.665387385448, 6851.955696328732, 120005.60201170924, 98167.41160254438,
    76472.0526321442, 56575.570250681274, 39689.74202007674, 26457.4179344202,
    16869.514269230876, 10369.967716745305, 6185.414086867278, 3599.0836240731496,
    35237.87035062025, 29469.46416516813, 23491.728527009203, 17785.76988233713,
    12758.346565487649, 8678.83248231447, 5630.3823712791545, 3511.4987843655877,
    2119.8114561428883, 1245.701436803505, 22187.53464815843, 18958.162955383403,
    15473.045903830529, 12015.121439793924, 8848.97644261989, 6174.299679877588,
    4095.5408011226937, 2601.237163950067, 1593.9786788896672, 948.5119301479799,
])

# C++ max model (Shear1h2hMax.so). Requires compute_lensing_2h = T.
CPP_MAX_MODEL = np.array([
    106957.62708374165, 89168.8513433637, 70872.76520986471, 53399.23828703865,
    37999.502584424125, 25622.34663983636, 16482.833648170894, 10217.57732838386,
    6135.886711374596, 3594.676371900488, 51265.849876841516, 44126.18737069989,
    36290.81396150937, 28314.99981245214, 20841.796933885882, 14486.954515302861,
    9565.197208047863, 6059.665999162044, 3705.7037127055733, 2202.270252123698,
    17096.645939253078, 15081.35159989328, 12747.638312404983, 10238.665375146971,
    7757.93450519037, 5539.106318825803, 3743.9903186340093, 2419.247873191509,
    1504.2891013447127, 906.4618778649381, 13482.256957956572, 12175.963224169152,
    10589.583226840188, 8794.087176852609, 6912.792809385156, 5118.731225221604,
    3574.5089116295644, 2374.3385411240547, 1511.2033853664718, 928.6622466330565,
    172638.19340442552, 143380.1621166534, 113526.73267765048, 85226.95568012915,
    60447.65225622269, 40639.945501338516, 26078.123472978063, 16131.657249760314,
    9670.432783229, 5665.608381985509, 76322.37174504978, 65478.03223646075,
    53665.395104147356, 41727.92773294821, 30615.607729268715, 21218.72336804949,
    13974.19208369337, 8833.630690981228, 5392.2296758060465, 3199.8389084490027,
    23223.47976610976, 20429.90904532617, 17216.603776242464, 13785.317633975994,
    10413.89980067798, 7414.799980706334, 4999.376895108041, 3223.538790926662,
    2000.7628958786493, 1203.8028203689853, 15773.46136176631, 14196.688983870681,
    12297.926684237109, 10168.505219796993, 7957.678335791926, 5867.063536251373,
    4080.658840843727, 2700.863132306381, 1713.681737166365, 1050.3003390106262,
    176768.09070190496, 145961.2743220072, 114901.7626384538, 85785.47097953156,
    60538.84919298257, 40521.795759622495, 25903.66737145875, 15972.745134201889,
    9549.703938611256, 5611.724711229513, 72794.35457173223, 62139.59708217075,
    50661.85582250414, 39187.80211406916, 28610.987643512894, 19741.574727734496,
    12950.82973767888, 8159.646169225127, 4966.929277984702, 2941.4480439388635,
    20374.855076931017, 17850.26008076189, 14974.57758339781, 11934.191349893501,
    8974.493931853365, 6362.966067293934, 4273.9591979556735, 2746.794859079739,
    1700.1192383241953, 1020.535494757458, 12152.07568066167, 10890.850156761517,
    9387.647761736544, 7720.591393506972, 6009.005302025304, 4407.003249896817,
    3050.1903593862453, 2010.0750136774325, 1270.5794131093783, 776.2354910605075,
])

# CUDA max model (Shear1h2hMaxGpu.so). Requires compute_lensing_2h = T.
CUDA_MAX_MODEL = np.array([
    106957.62708374216, 89168.85134336399, 70872.76520986503, 53399.238287038905,
    37999.50258442427, 25622.34663983637, 16482.833648170963, 10217.577328383893,
    6135.886711374616, 3594.676371900499, 51265.849876841676, 44126.18737070007,
    36290.813961509506, 28314.99981245221, 20841.796933885988, 14486.954515302901,
    9565.19720804791, 6059.665999162065, 3705.703712705582, 2202.270252123707,
    17096.645939253194, 15081.35159989332, 12747.638312404999, 10238.665375147008,
    7757.934505190371, 5539.106318825814, 3743.9903186340152, 2419.2478731915157,
    1504.2891013447158, 906.4618778649397, 13482.256957956592, 12175.963224169165,
    10589.583226840194, 8794.087176852634, 6912.792809385163, 5118.731225221612,
    3574.50891162957, 2374.3385411240583, 1511.203385366468, 928.6622466330584,
    172638.19340442627, 143380.16211665332, 113526.73267765081, 85226.95568012925,
    60447.65225622269, 40639.94550133864, 26078.123472978124, 16131.65724976034,
    9670.432783229051, 5665.608381985533, 76322.37174504991, 65478.03223646079,
    53665.39510414743, 41727.92773294829, 30615.607729268784, 21218.72336804952,
    13974.19208369342, 8833.630690981245, 5392.22967580608, 3199.838908449017,
    23223.479766109795, 20429.909045326214, 17216.603776242493, 13785.317633976041,
    10413.899800677995, 7414.799980706351, 4999.376895108051, 3223.5387909266724,
    2000.7628958786574, 1203.8028203689885, 15773.46136176633, 14196.688983870708,
    12297.926684237123, 10168.505219797007, 7957.678335791936, 5867.063536251381,
    4080.658840843724, 2700.8631323063823, 1713.681737166366, 1050.3003390106314,
    176768.09070190557, 145961.2743220075, 114901.76263845403, 85785.47097953182,
    60538.84919298266, 40521.7957596226, 25903.66737145877, 15972.745134201885,
    9549.703938611245, 5611.724711229525, 72794.35457173245, 62139.597082170956,
    50661.855822504214, 39187.80211406929, 28610.987643512923, 19741.57472773457,
    12950.829737678918, 8159.646169225134, 4966.9292779847, 2941.4480439388667,
    20374.855076931035, 17850.260080761946, 14974.57758339782, 11934.191349893523,
    8974.493931853385, 6362.966067293938, 4273.9591979556935, 2746.794859079747,
    1700.1192383241976, 1020.5354947574601, 12152.0756806617, 10890.850156761553,
    9387.64776173657, 7720.591393506989, 6009.0053020253235, 4407.003249896838,
    3050.1903593862444, 2010.075013677437, 1270.579413109378, 776.2354910605109,
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
        norm, ybar, mu = weights.moments_of(pf.y_of_lnM, ell_max=3)
        f_mis, tau_mis = dm.F_MIS_DEFAULT, dm.TAU_MIS_DEFAULT
        omega_m = self.source.scalar("cosmological_parameters", "omega_m")
        lob = np.asarray(dm.DEFAULT_LOB_CENTERS)

        vals = np.empty((12, R_PERP.size))
        for b in range(12):
            r_mis = tau_mis * float(dm.R_lambda(lob[b % lob.size]))
            vals[b] = evaluate_series(table, R_PERP, r_mis, norm[b], ybar[b],
                                      mu[b], f_mis=f_mis, rho_mult=omega_m,
                                      ell_max=2)
        np.testing.assert_allclose(vals.ravel(), CPP_RADIAL_SERIES, rtol=3e-4)

    def test_cpp_radial_series_matches_cpp_full_ltmz(self):
        # FINDING (real, at the project's standard 1e-3 tolerance -- see
        # docs/known_issues/radial_series_vs_full_ltmz_defect.md): radial_series's
        # raw DeltaSigma disagrees with the full_ltmz reference by
        # 56-86%, growing with richness bin (i.e. with mass).
        # nfw_profile_family.py hardcodes CONC = 4.0 for both the
        # profile shape AND its amplitude, instead of the per-sample
        # Child18 concentration full_ltmz/production interpolate (which
        # decreases with mass -- consistent with the worst mismatch at
        # the highest-richness bins). This is a real, previously
        # under-characterized finding: the strategy's own validator
        # only ever reported a shape-only, reported-not-asserted
        # comparison; the raw amplitude was never checked directly
        # against full_ltmz until this test. Kept at the project's
        # default tolerance rather than loosened, per policy -- a
        # failing test here is doing its job.
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
        # regeneration: docs/figs/real_pipeline_extract_max2h.ini.
        fast_cpp_proxy = self.source.array("shear1hmissel", "vals")
        self.assertTrue(np.all(CPP_MAX_MODEL >= fast_cpp_proxy * (1.0 - 1e-6)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
