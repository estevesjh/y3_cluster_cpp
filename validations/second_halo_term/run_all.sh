#!/bin/bash
# Two-halo term validation harness — full reproduction.
# Two interpreters, called by absolute path (no env activation needed):
#   conda  : pipeline env (camb 1.4.0, cluster_toolkit)
#   clenspy: CLensPy editable venv (clenspy, mcfit, pyccl, clmm)
# Cross-env exchange is plain-float64 .npz only (numpy 1.26 <-> 2.4 safe).
set -euo pipefail
cd "$(dirname "$0")"

CONDA=/opt/homebrew/Caskroom/miniforge/base/envs/y3cl_je_macos/bin/python
# -B: never write __pycache__ bytecode into the CLensPy repo
CLENSPY="/Users/jesteves/Documents/Dev/github/CLensPy/.venv/bin/python -B"

TAG="${1:-before}"   # before | after

echo "== 00 closed-form self-checks (both envs) =="
$CONDA   common/analytic_profiles.py
$CLENSPY common/analytic_profiles.py

echo "== 01 CAMB P(k,z) (generated once, shared) =="
$CONDA 01_make_pk_camb.py

echo "== 02 analytic chain bench: cluster_toolkit + DS methods =="
$CONDA 02_chain_bench_ct.py

echo "== 03 analytic chain bench: CLensPy transforms =="
$CLENSPY 03_chain_bench_clenspy.py

echo "== 04 fiducial reference: cluster_toolkit (per-z, converged) =="
$CONDA 04_reference_ct.py

echo "== 05 fiducial references: CLensPy + pyccl + clmm =="
$CLENSPY 05_reference_clenspy.py

echo "== 06 production evaluation (tag=$TAG) =="
if [ "$TAG" = "after" ]; then
    $CONDA 06_production_eval.py --tag after --method sandwich
    $CONDA 06_production_eval.py --tag after --method direct
else
    $CONDA 06_production_eval.py --tag before
fi

echo "== 07 compare, gates, figures, tables =="
$CONDA 07_compare.py
