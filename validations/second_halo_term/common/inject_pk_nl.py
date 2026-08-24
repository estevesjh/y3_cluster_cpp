"""CosmoSIS module (harness-local): publish matter_power_nl from the
harness's CAMB reference (outputs/pk_camb.npz, halofit).

Placed after cp_camb in the pipeline, this makes halo_model_cosmosis's
existing matter_power_nl branch take effect -- xi_nl, dSigma_hh and the
whole projection branch then run on halofit P(k) with zero production
code changes. This is the 'proposed fix' arm of the current-vs-proposed
comparison; with enabled=F it is a no-op (the 'current' linear-fallback
arm).
"""

import numpy as np
from cosmosis.datablock import option_section


def setup(options):
    enabled = options.get_bool(option_section, "enabled", default=True)
    npz_path = options.get_string(option_section, "npz")
    if not enabled:
        return {"enabled": False}
    d = np.load(npz_path, allow_pickle=False)
    return {"enabled": True, "k_h": d["k_h"], "z": d["z"],
            "p_k_nl": d["p_k_nl"]}


def execute(block, config):
    if not config["enabled"]:
        return 0
    block.put_grid("matter_power_nl", "z", config["z"],
                   "k_h", config["k_h"], "p_k", config["p_k_nl"])
    return 0


def cleanup(config):
    return 0
