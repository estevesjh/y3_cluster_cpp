# Observable strategies

The DES Y3 observable tree is organized as

```text
observable / strategy / implementation language
```

The strategy folder is the documentation boundary. Its README gives the
equations, the numerical recipe, the language-by-language execution steps,
the source-file inventory, and the precision/timing record.

| Observable | Meaning | Strategies present |
| --- | --- | --- |
| [Number counts](number_counts/README.md) | Expected clusters in observed richness and redshift bins | [`fast_mass`](number_counts/fast_mass/README.md), [`full_ltmz`](number_counts/full_ltmz/README.md) |
| [One-halo and traditional one-plus-two-halo shear](shear_1h2h/README.md) | Miscentred halo lensing and the optional max-model composition | [`fast_mass`](shear_1h2h/fast_mass/README.md), [`full_ltmz`](shear_1h2h/full_ltmz/README.md), [`radial_series`](shear_1h2h/radial_series/README.md) |
| [Projection shear](shear_projection/README.md) | Selection-affected correlated line-of-sight lensing | [`fast_mass`](shear_projection/fast_mass/README.md), [`full_ltmz`](shear_projection/full_ltmz/README.md) |

There is currently no `radial_series` implementation for number counts or
projection shear. Number counts has no radial operator, and projection shear
still has a coupled angular coordinate that has not been reduced to a
validated \(U_\ell\) table.
