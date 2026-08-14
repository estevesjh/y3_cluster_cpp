# Split `docs/sphinx-site` into three scoped PRs

**Status: done.** This PR (`docs/sphinx-site`, intra-fork
[estevesjh/y3_cluster_cpp#1](https://github.com/estevesjh/y3_cluster_cpp/pull/1))
now carries only the Sphinx site; the pipeline work moved to
[estevesjh/y3_cluster_cpp#7](https://github.com/estevesjh/y3_cluster_cpp/pull/7)
(`pipelines/des_y3`); the mf_tinker_cpp/macOS orphan bucket sits on
local branch `feature/mf-tinker-cpp`, not yet opened as a PR. See
"Outcome" at the bottom.

Both were originally opened as cross-fork PRs against
`marcpaterno/y3_cluster_cpp` (as `#4` and `#6`) and have since been
closed there in favor of intra-fork PRs on `estevesjh/y3_cluster_cpp`
— this fork isn't ready to propose anything upstream yet.

## Problem

`docs/sphinx-site` backs open PR
[marcpaterno/y3_cluster_cpp#4](https://github.com/marcpaterno/y3_cluster_cpp/pull/4).
It has grown to **85 commits / 165 files changed / 29,105 insertions** vs
`upstream/master`, plus 5 more commits sitting locally, unpushed. The diff
tangles three unrelated concerns, which makes the PR unreviewable as a
single unit: a reviewer either rubber-stamps everything or blocks the
Sphinx site on unresolved questions about GPU integration numerics that
have nothing to do with it.

Ground truth (not GitHub's cached PR summary, which undercounts):

```
git diff upstream/master...origin/docs/sphinx-site --stat
# 165 files changed, 29105 insertions(+), 85 deletions(-)
```

## The three tangled concerns

| Bucket | Files | Content |
|---|---|---|
| **Docs (Sphinx site)** | 55 | `docs/source/**`, `docs/Makefile`, `docs/requirements.txt`, `.readthedocs.yaml`, `docs/figs/*`. No runtime behavior change. |
| **`des_y3` pipeline work** | 67+ | `src/pipelines/des_y3/**`, `src/models/n_operator_sel_gl_t.hh`, `src/modules/num_counts_sel/*`, `sel_function.py`, 20 new tests, 13 new `data/radial_series/*` tables. Real feature work: full_ltmz/fast_mass/radial_series backends, fixed-GL evaluators, cross-backend consistency tests. |
| **Orphan: `mf_tinker_cpp` + macOS build** | 27 (5 commits, unpushed) | FFTlog/Chebyshev HMF replacement module + macOS CPU-only build support. Unrelated to both of the above. |

`CLAUDE.md` and `BUILDING.md` are each touched by commits in all three
buckets, so a clean per-commit split isn't possible without conflicts —
history is genuinely interleaved (e.g. "Add C++ full_ltmz shear backend"
sits a few commits from "Document the reorganized DES Y3 pipeline").

PR #4 has **zero reviews/comments** as of this writing, so rewriting its
history is safe — nothing is invalidated by a force-push.

## Why split (best-practice rationale)

- **Reviewability**: a docs-only PR needs no GPU/CUDA CI and no domain
  review of unit conventions; the pipeline PR needs exactly that.
- **Bisectability**: a future des_y3 regression should `git bisect` to a
  pipeline commit, not scroll past 55 unrelated doc commits.
- **Independent merge cadence**: docs merges the moment
  `sphinx-build -W` passes; the pipeline PR waits on real review of the
  fixed-GL math and new tests. Coupling them forces both to wait on the
  slower one.
- **History quality over history preservation**: most of the 85 commits
  are process narration ("Explain the cost gap...", "Revise... planned,
  not dismissed", "Record approval of..."). Given the `CLAUDE.md`/
  `BUILDING.md` overlap already rules out a faithful per-commit
  cherry-pick, squashing into a handful of thematic commits per branch
  serves future `git blame` better than preserving the raw sequence.

## Steps

1. **Peel off the orphan bucket** → new branch `feature/mf-tinker-cpp` off
   `master`; move the 5 unpushed commits (mf_tinker_cpp module + macOS
   build support) there. Own PR, opened later, independent of both docs
   and des_y3 review.
2. **Rebuild `docs/sphinx-site` off `master`**: checkout the docs-only
   paths from the current tip, squash into a handful of thematic commits
   (e.g. "Add Sphinx documentation site", "Reorganize testing docs",
   "Cite Costanzi 2026 / adopt paper language"). Force-push over the
   existing branch — this becomes the new content of PR #4.
3. **Build `pipelines/des_y3` off `master`**: checkout
   `src/pipelines/des_y3/**`, the GL evaluator + `num_counts_sel` files,
   tests, and data tables; squash into logical commits (e.g. "Add
   des_y3 namespace + full_ltmz reference", "Add fast_mass C++/CUDA
   backends", "Add radial_series backend + tables", "Add cross-backend
   consistency tests"). Open as a **new PR**.
4. **Reconcile shared files by hand**: `CLAUDE.md`/`BUILDING.md` get
   their pipeline-relevant hunks in the pipelines branch and their
   doc-relevant hunks in the docs branch — split by content, not by
   commit.
5. **Verify each branch independently**:
   - docs branch → `sphinx-build -W -b html docs/source docs/build/html`
   - pipelines branch → CPU-only build + relevant `ctest` targets
6. **Push**: force-push the rebuilt `docs/sphinx-site` (PR #4 updates in
   place); push `pipelines/des_y3` and open its PR; push
   `feature/mf-tinker-cpp` and open its PR when ready.

## Decisions made

- Orphan bucket (`mf_tinker_cpp` + macOS build): **own branch/PR**, not
  folded into either docs or des_y3.
- `docs/sphinx-site` / PR #4: **force-push the squashed rebuild** — safe,
  no existing reviews to invalidate.

## A fourth bucket, found mid-split

`docs/sphinx-site` had branched off local `master`, which was itself 2
commits ahead of `upstream/master` (`NumCountsSel`/`Shear1hMisSel`
fixed-GL evaluator work — a different, already-in-progress line of
work, unrelated to the PR #3 author). Those 2 commits' files
(`src/models/n_operator_sel_gl_t.hh`, `src/modules/num_counts_sel/*`)
were riding along in the diff purely as inherited baseline, untouched
by any docs/sphinx-site commit — confirmed by an empty
`git diff master tmp/full-tip -- <those paths>`. **Excluded from both
new branches**; they stay on local `master`, unpushed, for you to PR
separately whenever.

## Outcome

- All branch-building happened in an isolated git worktree
  (`../y3_cluster_cpp_split_backup`, branch `tmp/full-tip` as the
  combined source) so the main working tree and its running CosmoSIS
  job were never touched.
- `feature/mf-tinker-cpp` (4 commits, off `upstream/master`): clean,
  self-contained, no `des_y3`/docs leakage.
- `docs/sphinx-site` (4 commits, off `upstream/master`): verified with
  `sphinx-build -W -b html docs/source docs/build/html` — build
  succeeded. Force-pushed to `origin/docs/sphinx-site`. Originally
  updated PR #4 on `marcpaterno/y3_cluster_cpp` in place; that cross-fork
  PR is now closed in favor of the pre-existing intra-fork
  [estevesjh/y3_cluster_cpp#1](https://github.com/estevesjh/y3_cluster_cpp/pull/1),
  which tracks the same branch.
- `pipelines/des_y3` (3 commits, off `upstream/master`): all
  `add_subdirectory`/test-file path references verified to resolve, all
  new Python files syntax-checked. Pushed and opened first as PR #6 on
  `marcpaterno/y3_cluster_cpp`, then closed there in favor of intra-fork
  [estevesjh/y3_cluster_cpp#7](https://github.com/estevesjh/y3_cluster_cpp/pull/7).
  Full `ctest`/GPU build still needed on a Perlmutter node before merge.
