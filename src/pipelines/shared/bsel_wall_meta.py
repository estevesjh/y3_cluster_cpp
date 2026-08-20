"""CosmoSIS module: publish the b_sel_marg wall metadata
(lambda_bin, zo_low, zo_high) into the b_sel_marg_P1/_I1/_J sections.

The C++ evaluator template (CosmoSISScalarEvaluatorModule) publishes only
'vals'; the wall geometry lives in the [b_sel_marg] ini section and never
enters the datablock. The refactored consumers
(src/pipelines/shared/datablock_models.py BSelWallVector.from_source)
expect it there. This module closes that contract gap non-invasively:
run it right after b_sel_marg and give it the SAME wall vectors as
[b_sel_marg]. Resolution of issue estevesjh/y3_cluster_cpp#10 (option 2
of the three considered there): a pipeline module rather than a C++
template change, keeping src/utils immutable.
"""

import numpy as np
from cosmosis.datablock import option_section

SECTIONS = ("b_sel_marg_P1", "b_sel_marg_I1", "b_sel_marg_J")


def setup(options):
    lam = np.asarray(options[option_section, "lambda_bin"], dtype=np.int32)
    zo_low = np.asarray(options[option_section, "zo_low"], dtype=float)
    zo_high = np.asarray(options[option_section, "zo_high"], dtype=float)
    assert lam.shape == zo_low.shape == zo_high.shape
    return {"lambda_bin": lam, "zo_low": zo_low, "zo_high": zo_high}


def execute(block, config):
    for sec in SECTIONS:
        block[sec, "lambda_bin"] = config["lambda_bin"]
        block[sec, "zo_low"] = config["zo_low"]
        block[sec, "zo_high"] = config["zo_high"]
    return 0


def cleanup(config):
    return 0
