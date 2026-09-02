# Testing inventory

This page indexes the tests by source folder. Each inventory records the CTest
target, the test source, the implementation or script under test, what the
test covers, and its current status.

Run the configured suite with:

```bash
ctest -j 6 --output-on-failure
```

Use `ctest -N` for the exact target list. Default relative tolerance is
`1e-3`. Two known-failing tests are deliberate (real defects, not test
bugs):
[`radial_series_vs_full_ltmz_defect.md`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/docs/known_issues/radial_series_vs_full_ltmz_defect.md),
[`nfw_dsigma_mis_defect.md`](https://github.com/estevesjh/y3_cluster_cpp/blob/docs/sphinx-site/docs/nfw_dsigma_mis_defect.md).

## Folder inventories

```{toctree}
:maxdepth: 1

testing/src_pipelines_des_y3
testing/src_modules
```

Status labels mean:

- **Passing** — last recorded configured run passed.
- **Known failing** — intentionally exposes an unresolved defect; see the
  linked issue note.
- **Characterization** — asserts or records currently observed behavior,
  including a known numerical defect, rather than declaring that behavior
  scientifically correct.
- **Disabled/diagnostic** — not part of the configured CTest suite.

## External validations

Cross-code comparisons, reference benchmarks, and production diagnostics
live in the sibling [`scratchReports`](https://github.com/estevesjh/scratchReports)
repository under `y3_cluster_cpp/`. These are not unit tests — they validate
against independent reference implementations (cluster_toolkit, CLensPy, pyccl,
clmm) and produce LaTeX validation reports. See the repo's README for setup
and execution.
