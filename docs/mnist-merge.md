# MNIST Branch Reconciliation

**Date:** 2026-05-29
**Author:** Cagdas Ucar
**Status:** Pre-implementation
**Next:** [inference-level.md](./inference-level.md)

---

## Why

The `mnist` branch has accumulated substantial work alongside experimental v1 moment-neuron scaffolding. Before any new structural work lands on top of it, the keepable parts must be merged into `dev`. Merging late would compound conflicts with later phases.

This is a prerequisite for everything downstream — inference-scope experimentation, spatial processing, neuron reuse.

---

## Phase 1 — Branch Reconciliation (mnist → dev)

### Steps

1. List every file changed on `mnist` vs `main`. Classify each:
   - **Permanent — core** — brain/runtime changes that should ship regardless of later phases (e.g. bug fixes, perf improvements, infra cleanup). Merge to `dev`.
   - **Permanent — apps** — MNIST encoder, harness, jobs that will still be useful for later MNIST validation. Merge to `dev`.
   - **Experimental v1** — moment-neuron v1 scaffolding superseded by the spatial-processing design. Leave on `mnist`; do not merge.
   - **Throwaway** — debug prints, parameter-sweep scripts, ad-hoc harnesses. Leave on `mnist` or delete.
2. For every "Permanent — core" candidate, assess impact on the stocks pipeline before merging. Anything that changes stocks behavior gets called out explicitly so we know what the post-merge baseline is.
3. Open the merge into `dev` as a reviewable PR with the classification table in the description.
4. Tag the pre-merge tip of `mnist` (e.g. `mnist-v1-final`) so v1 MNIST experiments stay reproducible.
5. After merge, run the stocks pipeline on `dev` and record the new baseline.

### Known permanent-core change to land here

**Static forget rate.** Replace the existing level-dependent forget rate with a single global static forget rate. Experimentation on the `mnist` branch found no meaningful accuracy difference between level-dependent and static decay, and static is simpler and more biologically defensible (no reason higher-level neurons should follow different decay rules). This drops one parameter and removes a level dependency that would otherwise complicate downstream removal of intrinsic neuron levels.

### Acceptance

- Stocks directional accuracy on merged `dev` ≥ current `main` baseline (within ±1% noise).
- Brain unit tests pass on `dev`.
- Classification table is committed alongside the merge so future readers know what intentionally stayed on the branch.
- Static forget rate is in effect on `dev`; no `level_forget_rate` table or per-level decay logic remains.

### Notes

- All subsequent phases happen on `dev` or branches off `dev`. `mnist` becomes archival.
- If the merge surfaces work that's worth keeping but risky, land it as a follow-up PR after this phase closes — keep the initial merge focused on no-regression changes.
