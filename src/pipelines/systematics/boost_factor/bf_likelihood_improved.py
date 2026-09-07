# Author: Arwa (improved version)

import numpy as np
import os
from pathlib import Path

# CosmoSIS import - only needed when running within CosmoSIS
try:
    from cosmosis.datablock import names
    HAS_COSMOSIS = True
except ImportError:
    HAS_COSMOSIS = False


# =============================================================================
# BOOST FACTOR MODEL
# =============================================================================

def boost_factor_model(R, rs, b0):
    """
    Parameters
    ----------
    R : array_like
        Radial distances (in Mpc/h or consistent units with rs)
    rs : float
        Scale radius parameter
    b0 : float
        Amplitude parameter

    Returns
    -------
    B : ndarray
        Boost factor values at each radius
    """
    x = np.atleast_1d(R / rs).astype(float)
    B = np.zeros_like(x, dtype=float)

    # Define tolerance for x ~ 1 (removable singularity)
    tol = 1e-6

    # Masks for different regimes
    mask_near1 = np.abs(x - 1) < tol
    mask_gt1 = (x > 1) & ~mask_near1
    mask_lt1 = (x < 1) & ~mask_near1

    # x > 1: arctan regime
    if np.any(mask_gt1):
        x_gt1 = x[mask_gt1]
        sqrt_term = np.sqrt(x_gt1**2 - 1)
        fx = np.arctan(sqrt_term) / sqrt_term
        B[mask_gt1] = 1 + b0 * (1 - fx) / (x_gt1**2 - 1)

    # x < 1: arctanh regime
    if np.any(mask_lt1):
        x_lt1 = x[mask_lt1]
        sqrt_term = np.sqrt(1 - x_lt1**2)
        fx = np.arctanh(sqrt_term) / sqrt_term
        B[mask_lt1] = 1 + b0 * (1 - fx) / (x_lt1**2 - 1)

    # x ~ 1: use the analytic limit (b0 + 3) / 3
    if np.any(mask_near1):
        B[mask_near1] = (b0 + 3) / 3

    # Handle any remaining NaN/inf values
    B = np.where(np.isnan(B) | np.isinf(B), (b0 + 3) / 3, B)

    return B


# =============================================================================
# DATA LOADING - DES Y1 ONLY
# =============================================================================

class BoostFactorData:
    """Container for boost factor data from a single bin."""

    def __init__(self, R, data_vector, covariance, richness_bin, redshift_bin):
        self.R = R
        self.data_vector = data_vector
        self.covariance = covariance
        self.inv_cov = np.linalg.inv(covariance)
        self.richness_bin = richness_bin
        self.redshift_bin = redshift_bin
        self.n_points = len(R)

    @property
    def bin_label(self):
        return f"l{self.richness_bin}_z{self.redshift_bin}"


def load_y1_data(data_path, richness_bin, redshift_bin, n_points=8):
    """
    Parameters
    ----------
    data_path : str
        Path to the directory containing the data files
    richness_bin : int
        Richness bin index (0-3)
    redshift_bin : int
        Redshift bin index (0-2)
    n_points : int
        Number of radial points to use (default: 8)

    Returns
    -------
    BoostFactorData
        Data container with loaded data
    """
    data_file = os.path.join(
        data_path,
        f"full-unblind-v2-mcal-zmix_y1clust_l{richness_bin}_z{redshift_bin}_zpdf_boost.dat"
    )
    cov_file = os.path.join(
        data_path,
        f"full-unblind-v2-mcal-zmix_y1clust_l{richness_bin}_z{redshift_bin}_zpdf_boost_cov.dat"
    )

    R, data_vector, sigma_B = np.genfromtxt(data_file, unpack=True)
    covariance = np.genfromtxt(cov_file)

    # Truncate to n_points
    R = R[:n_points]
    data_vector = data_vector[:n_points]
    covariance = covariance[:n_points, :n_points]

    return BoostFactorData(R, data_vector, covariance, richness_bin, redshift_bin)


def discover_y1_bins(data_path):
    """
    Automatically discover available Y1 bins in the data directory.

    Parameters
    ----------
    data_path : str
        Path to the data directory

    Returns
    -------
    list of tuple
        List of (richness_bin, redshift_bin) tuples
    """
    bins = []
    path = Path(data_path)

    pattern = '*_l*_z*_zpdf_boost.dat'
    for f in path.glob(pattern):
        name = f.stem
        parts = name.split('_')
        l_val, z_val = None, None
        for p in parts:
            if p.startswith('l') and len(p) > 1 and p[1:].isdigit():
                l_val = int(p[1:])
            if p.startswith('z') and len(p) > 1 and p[1:].isdigit():
                z_val = int(p[1:])
        if l_val is not None and z_val is not None:
            bins.append((l_val, z_val))

    return sorted(bins)


# =============================================================================
# COSMOSIS INTERFACE - Y1 DATA
# =============================================================================

def setup(options):
    """
    CosmoSIS setup function - loads Y1 data and prepares configuration.

    Options (from .ini file):
    - data_path: Path to Y1 data directory
    - richness_bins: Comma-separated list or 'all'
    - redshift_bins: Comma-separated list or 'all'
    - n_radial_points: Number of radial points to use (default: 8)
    """
    section = "boost_factor_likelihood"

    data_path = options.get_string(section, "data_path")
    n_radial_points = options.get_int(section, "n_radial_points", default=8)

    # Determine which bins to use
    richness_bins_str = options.get_string(section, "richness_bins", default="all")
    redshift_bins_str = options.get_string(section, "redshift_bins", default="all")

    if richness_bins_str == "all" or redshift_bins_str == "all":
        available_bins = discover_y1_bins(data_path)
        if richness_bins_str == "all":
            richness_bins = sorted(set(b[0] for b in available_bins))
        else:
            richness_bins = [int(x.strip()) for x in richness_bins_str.split(',')]
        if redshift_bins_str == "all":
            redshift_bins = sorted(set(b[1] for b in available_bins))
        else:
            redshift_bins = [int(x.strip()) for x in redshift_bins_str.split(',')]
    else:
        richness_bins = [int(x.strip()) for x in richness_bins_str.split(',')]
        redshift_bins = [int(x.strip()) for x in redshift_bins_str.split(',')]

    # Load data for all requested bins
    bin_data = {}
    for l in richness_bins:
        for z in redshift_bins:
            try:
                data = load_y1_data(data_path, l, z, n_radial_points)
                bin_data[(l, z)] = data
                print(f"Loaded data for bin l={l}, z={z} ({data.n_points} points)")
            except Exception as e:
                print(f"Warning: Could not load bin l={l}, z={z}: {e}")

    config = {
        'bin_data': bin_data,
        'bins': list(bin_data.keys())
    }

    print(f"Successfully loaded {len(bin_data)} bins")
    print("Data format: B (boost factor) - DES Y1")
    return config


def execute(block, config):
    """
    CosmoSIS execute function - computes likelihood for all bins.
    """
    bin_data = config['bin_data']

    total_log_L = 0.0

    for (l, z), data in bin_data.items():
        # Read parameters for this bin from datablock
        param_suffix = f"l{l}_z{z}"

        try:
            logrs = block["boost_factor_params", f"logrs_{param_suffix}"]
            logb0 = block["boost_factor_params", f"logb0_{param_suffix}"]
        except Exception:
            # Fallback to old naming convention
            logrs = block["Boost_Factor_Model_Values", f"logrs_{l}{z}"]
            logb0 = block["Boost_Factor_Model_Values", f"logb0_{l}{z}"]

        rs = 10**logrs
        b0 = 10**logb0

        # Compute model prediction (boost factor B)
        B_model = boost_factor_model(data.R, rs, b0)

        # Y1 data is already in B units - compare directly
        diff = B_model - data.data_vector
        chisq = np.dot(diff, np.dot(data.inv_cov, diff))

        log_L = -0.5 * chisq
        total_log_L += log_L

        # Store individual bin likelihoods for diagnostics
        block["boost_factor_diagnostics", f"chisq_{param_suffix}"] = chisq
        block["boost_factor_diagnostics", f"logL_{param_suffix}"] = log_L

    # Store total likelihood
    block["likelihoods", "boost_factor_likelihood_like"] = total_log_L

    return 0


# =============================================================================
# STANDALONE TESTING UTILITIES
# =============================================================================

def compute_likelihood_standalone(R, data_vector, covariance, rs, b0):
    """
    Compute the log-likelihood for given parameters (for testing outside CosmoSIS).

    Parameters
    ----------
    R : array_like
        Radial distances
    data_vector : array_like
        Observed boost factor B values (Y1 format)
    covariance : array_like
        Covariance matrix
    rs : float
        Scale radius
    b0 : float
        Amplitude

    Returns
    -------
    log_L : float
        Log-likelihood value
    chisq : float
        Chi-squared value
    model : ndarray
        Model predictions
    """
    inv_cov = np.linalg.inv(covariance)
    B_model = boost_factor_model(R, rs, b0)

    diff = B_model - data_vector
    chisq = np.dot(diff, np.dot(inv_cov, diff))
    log_L = -0.5 * chisq

    return log_L, chisq, B_model


def generate_values_ini(bins, output_file, prior_range=(-1.0, 0.0, 1.0)):
    """
    Generate a CosmoSIS values .ini file for the given bins.

    Parameters
    ----------
    bins : list of tuple
        List of (richness_bin, redshift_bin) tuples
    output_file : str
        Output file path
    prior_range : tuple
        (min, start, max) for log parameters
    """
    lines = ["[boost_factor_params]"]
    lines.append("# Parameter file for boost factor model - DES Y1")
    lines.append("# 4 richness bins (l=0-3) x 3 redshift bins (z=0-2) = 12 bins")
    lines.append("# Format: logrs_lX_zY = min start max")
    lines.append("")

    for l, z in sorted(bins):
        suffix = f"l{l}_z{z}"
        lines.append(f"logrs_{suffix} = {prior_range[0]} {prior_range[1]} {prior_range[2]}")
        lines.append(f"logb0_{suffix} = {prior_range[0]} {prior_range[1]} {prior_range[2]}")

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Generated values file: {output_file}")


if __name__ == "__main__":
    print("Boost Factor Likelihood Module - DES Y1")
    print("=" * 45)

    # Test the model function
    print(f"Model test: B(R=1, rs=1, b0=0.3) = {boost_factor_model(np.array([1.0]), 1.0, 0.3)[0]:.4f}")
    print(f"Expected at x=1: (b0+3)/3 = {(0.3+3)/3:.4f}")
    print("\nY1 Data: 4 richness bins (l=0-3) x 3 redshift bins (z=0-2) = 12 bins")
    print("Data format: Boost factor B(R)")
