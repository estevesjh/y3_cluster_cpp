# Improved Boost Factor Module - DES Y1

This folder contains the improved boost factor likelihood code for DES Year 1 data with multi-bin automation.

**Reference**: McClintock et al. (2019), arXiv:1805.00039


## Relationship to the main pipeline

**This whole folder exists only to *compute* the boost-factor calibration
(rs, b0) once, offline, against the real DES Y1 data.** The main GPU
pipeline (`shearTot_pipeline.ini`) does not
run any fit itself -- it only reads the already-fitted `rs_l*_z*` /
`b0_l*_z*` numbers as static values from `values_gpu.ini`'s
`[boost_factor]` section, via `apply_boost_factor.py`.
contact Arwa Abdulghafour for more info. 

## Files

| File | Description |
|------|-------------|
| `apply_boost_factor.py` | **Used by the main pipeline.** CosmoSIS module that publishes B(R) per bin from the static rs/b0 values in `values_gpu.ini` -- does not fit anything itself. |
| `fit_y1_bins.py` | Calibration script: chi-square fit of `boost_factor_model()` against the real Y1 data for all 12 bins. This is what actually produced the `rs_l*_z*`/`b0_l*_z*` numbers now sitting in `values_gpu.ini`. Run offline, not part of the main pipeline. |
| `bf_likelihood_improved.py` | Shared likelihood/model code used by both `apply_boost_factor.py` and `fit_y1_bins.py`. |
| `bf_pipeline_improved.ini` | Standalone CosmoSIS pipeline for testing/re-running the calibration fit, separate from the main GPU pipeline. |
| `bf_values_all_bins.ini` | Parameter priors for all 12 bins, used by the standalone calibration pipeline above. |
| `test_bf_likelihood.ipynb` | Jupyter notebook to test the module. |

## Usage

### Testing locally (without CosmoSIS):
```python
from bf_likelihood_improved import (
    boost_factor_model,
    load_y1_data,
    discover_y1_bins,
    compute_likelihood_standalone
)

# Load data
data = load_y1_data('path/to/y1/profiles', l=0, z=0, n_points=8)

# Compute likelihood
log_L, chi2, model = compute_likelihood_standalone(
    data.R, data.data_vector, data.covariance,
    rs=1.0, b0=0.3
)
```

### Running with CosmoSIS:
```bash
cosmosis bf_pipeline_improved.ini
```

## Data Files (Y1)

Expected file format:
- `full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost.dat` - R, B, sigma_B
- `full-unblind-v2-mcal-zmix_y1clust_l{l}_z{z}_zpdf_boost_cov.dat` - Covariance matrix
