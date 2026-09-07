# Halo Mass Function

`Fortran 90` · `CosmoSIS Standard Library` · `Halo mass function` · module `MfTinker` · `155 ms/sample`

Computes the Tinker et al. (2008) halo mass function from the linear
matter power spectrum. Every cluster-population integral downstream
(number counts, halo shear, selection-bias operators, projection lensing)
weights by its `dndlnmh` output.

## Script

- Source: [`mass_function/mf_tinker/`](https://github.com/joezuntz/cosmosis-standard-library/tree/main/mass_function/mf_tinker)
  — `tinker_module.F90`, `compute_mf_tinker.f90`, `mf_tinker.f90` and
  supporting Fortran-90 sources (based on Komatsu's CRL).
- Compiled library loaded by CosmoSIS:
  `${COSMOSIS_STANDARD_LIBRARY}/mass_function/mf_tinker/tinker_mf_module.so`.

## Numerical framework

Tinker et al. (2008) multiplicity function
$f(\sigma)$ with $\Delta = 200\bar\rho_m$:

$$\frac{dn}{d\ln M} = f(\sigma)\,\frac{\bar\rho_m}{M}\,
\left|\frac{d\ln\sigma^{-1}}{d\ln M}\right| ,$$

with $\sigma(M, z)$ from the input linear spectrum.

```{warning}
**Mass-axis convention.** `m_h` is *not* $M h^{-1} M_\odot$: the CosmoSIS
Tinker module tabulates against $M/(\Omega_m - \Omega_\nu)$. All C++
consumers query through `HMF_t`, which applies the
$\ln(\Omega_m - \Omega_\nu)$ shift internally; a raw `lnM` lookup lands
$\approx 0.6$ dex off and silently returns wrong HMF values. Python
consumers (`bsel.py`) rescale the axis explicitly.
```

Longer discussion: {doc}`../numerics/index` (unit and convention traps).

## CosmoSIS setup

```ini
[MfTinker]
file = ${COSMOSIS_STANDARD_LIBRARY}/mass_function/mf_tinker/tinker_mf_module.so
redshift_zero = 0
feedback = 0
matter_power_lin_version = 2
```

- Ordering: after `cp_camb` (needs the linear power grid).
- `matter_power_lin_version = 2` makes it read the **CDM+baryon**
  (no-neutrino) spectrum `cdm_baryon_power_lin`, the correct prescription
  for halo statistics with massive neutrinos — `cp_camb` must publish that
  section (it does, from its `linear_nonu_pk_path` emulator).

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `redshift_zero` | 1 = output only the $z=0$ mass function; 0 = one per redshift slice | — | 0 |
| `feedback` | verbosity (0 silent) | — | 0 |
| `matter_power_lin_version` | 2 = read `cdm_baryon_power_lin` instead of `matter_power_lin` | — | 2 |

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `cdm_baryon_power_lin/k_h` | wavenumber grid | $h/\mathrm{Mpc}$, `(506,)` | `cp_camb` |
| `cdm_baryon_power_lin/z` | redshift grid | `(50,)` | `cp_camb` |
| `cdm_baryon_power_lin/p_k` | CDM+baryon linear power | $(\mathrm{Mpc}/h)^3$, `(50, 506)` | `cp_camb` |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `mass_function/m_h` | mass grid, in the CosmoSIS convention $M\,\Omega_m^{-1}\,h^{-1} M_\odot$ (see warning) | `(969,)` | `NumCountsSijGl`, `Shear1hGl`, `b_sel_marg`, `ShearPrjGl`, `bsel` |
| `mass_function/z` | redshift grid | `(50,)` | same |
| `mass_function/dndlnmh` | $dn/d\ln M$ | $h^3\,\mathrm{Mpc}^{-3}$, `(50, 969)` | same |
| `mass_function/r_h`, `dndlnrh` | radius-space equivalents | $h^{-1}$Mpc | — (unused here) |

