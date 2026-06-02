# Short-Term Roadmap

Each section below is one step. They execute in order — steps 1–4 land the architecture, steps 5–6 are the MNIST validation milestones, steps 7–10 are follow-on work.

---

## 1. Naive MNIST

See **[mnist-merge.md](./mnist-merge.md)**.

Reconcile the long-running `mnist` branch into `dev`. This brings in the **sensory-only (Naive Bayes) MNIST app** — the degenerate, pre-spatial-processing iteration. See that doc for why it is structurally Naive Bayes and why its accuracy is capped accordingly.

---

## 2. Inference Level Experiment

See **[inference-level.md](./inference-level.md)**.

Pick the inference scope rule (`base` / `same-level` / `all-levels`) via an experiment on the stocks pipeline. Decision propagates into spatial processing.

---

## 3. Spatial Processing

See **[spatial-processing.md](./spatial-processing.md)**.

Add `process_spatial` ahead of `process_temporal`. Remove intrinsic neuron levels (level becomes per-frame activation state). d=0 connections, spatial wavefront, error-driven correction minting. Validates on MNIST single-frame and on stocks. **This is the workstream that begins relaxing the Naive Bayes independence assumption** by manufacturing conjunctive features.

---

## 4. Neuron Re-use

See **[neuron-reuse.md](./neuron-reuse.md)**.

Reverse inference index, reuse lookup in the correction path, transfer-learning validation, full-pipeline stocks integration, class-neuron generalization tuning. Allocates capacity onto the error manifold (residual-fitting).

---

## 5. Vanilla MNIST

Vanilla 10-way digit classification with the completed architecture. Runs only after steps 1–4 land. Confirms the architecture handles vision at all before tackling continual learning.

### What's already in place from step 1

The channel layout, encoder, episode shape, training/eval harness, and hyperparameter starting points were stood up in step 1 as the sensory-only Naive Bayes app — see the **Naive MNIST app** section in [mnist-merge.md](./mnist-merge.md) for retinotopic channels, single-frame episode structure, phased quantization (binary → 4/8 → 16), shared action neurons, and compute notes. This step does **not** change any of that. What changes is the brain underneath: spatial processing (step 3) now manufactures inter-channel connections at d=0, and neuron reuse (step 4) reallocates capacity onto the error manifold. Same encoder, same harness, same channels — the architecture is no longer degenerate.

### The baseline to beat: Naive Bayes

The sensory-only app merged in step 1 *is* Naive Bayes — independent per-pixel voting, no joint structure (see mnist-merge.md). That sets the explicit bar for the full architecture:

* **Naive Bayes (independent pixels): ~83–84% test on full 28×28.** This is the floor and the "did we just reimplement NB?" check. The full architecture must clear this *using a different mechanism* (no backprop, no labels, demand-driven, gain from conjunctive features and reuse) for the result to mean anything.
* **Logistic regression / linear classifier: ~92%.** The next rung — pixel-based but jointly weighted. Beating NB but not this means a little co-occurrence is being captured but nothing strongly spatial.
* **k-NN: ~97%.** Pure template matching; shows how much signal is in the pixels when configuration is respected.
* **Simple MLP / small CNN: 98–99%+.** The gradient-trained ceiling — not the target for a label-free, no-backprop voting system, but the reference for where the signal tops out.

The interesting result for Robot Brain is not beating a CNN; it is **matching or beating a jointly-trained linear model (~92%) without joint training, labels, or backprop** — with the climb from the NB floor coming mechanism by mechanism as the independence assumption dissolves. Each architectural addition (spatial processing, then reuse) should push accuracy up this ladder while preserving the no-backprop / no-label properties. Don't over-anchor on NB's exact number; let an NB run on *identical* preprocessing be the apples-to-apples reference.

### Why MNIST as the validation target

MNIST is the most widely recognized benchmark in machine learning. We use it twice, in sequence, to make two distinct architectural claims:

1. **Vanilla MNIST** (this step) — demonstrates that one prediction-only architecture handles stocks, text, and vision with zero modifications. Validates the "one substrate, multiple domains" claim.
2. **Split-MNIST (class-incremental)** (next step) — demonstrates continual learning without replay buffers, task IDs, regularizers, or any of the workarounds gradient-based models require. The headline result for external positioning.

The deeper claim Split-MNIST validates: **Robot Brain is constitutionally immune to catastrophic forgetting.** Neurons aren't overwritten by gradient updates; they're added and decayed independently. Transformers and MLPs trained sequentially on disjoint class subsets collapse to ~20% accuracy on old tasks (Hsu et al. 2018; van de Ven & Tolias 2019). This architecture should retain old-task accuracy by construction — patterns formed for digits 0/1 are not modified when the system later sees digits 2/3, because their context fingerprints don't match.

### What spatial processing adds at this step

With `process_spatial` in place, connection formation is no longer just sensory → action. Within the single image-frame, every pixel channel observes what its neighbors are concurrently firing at d=0, and inter-channel connections form on the fly. The system learns:

* **Local spatial correlations**: pixel (14,14) being dark while pixel (14,15) is also dark is a learned association, not a geometric prior.
* **Global digit signatures**: the constellation of pixel activations that characterize each digit class.
* **Discriminative features**: through reward, the system reinforces connections whose activation patterns reliably predict specific digits.

The brain discovers that certain combinations of pixel values across specific spatial positions predict specific digits — without ever being told that pixels are arranged in a grid, or that adjacent pixels tend to co-vary. This is the conjunctive-feature mechanism the NB ladder is built to detect.

### Phase 1 success criteria

* **The bar is Naive Bayes (~83–84% at 28×28), not chance.** The sensory-only merge app from step 1 already reaches the NB ceiling by construction. Phase 1 runs the *completed* architecture (post spatial-processing and reuse), so its job is to clear the NB floor using the conjunctive-feature mechanism — anything at or below NB means the spatial/reuse machinery is not yet contributing joint structure and should be debugged before adding precision.
* **Phase A (binary), gate**: must clear NB at matched preprocessing (28×28 binary NB run from step 1 is the apples-to-apples reference). Failing to clear NB at binary is the architectural debug signal — adding bucket precision will not fix it.
* **Phases B/C, target**: climb the ladder. **Matching or beating the linear-classifier rung (~92%)** at the optimal bucket count is the headline Phase 1 result — that's the "joint structure without joint training" claim. Pushing into the k-NN range (~97%) is stretch.
* Training accuracy converges to >95% with sufficient repetition at the optimal quantization level.
* No architectural changes vs stock/text channels — same brain code, same connection mechanism, just more channels.
* Inter-channel connections demonstrably encode spatial structure (inspectable: which pixel positions form strong connections should roughly correspond to spatial proximity and shared digit-class membership).
* The quantization-vs-accuracy curve is documented as an empirical result characterizing the architecture's sensory resolution tradeoff — paired with the NB-only curve from step 1 to isolate the architectural contribution from the resolution contribution.

This is not an attempt to beat CNNs. It is a demonstration that a single prediction-only architecture, designed for temporal sequences, can learn visual recognition through spatial co-occurrence across parallel channels — without any vision-specific components. The benchmark exists to make the architectural claim legible to the ML community using a universally understood task.

---

## 6. Split MNIST

Class-incremental continual learning. The **headline experiment for external positioning**. Runs only after Phase 1 confirms vanilla MNIST works.

### Protocol

Standard class-incremental Split-MNIST as defined in the continual learning literature (van de Ven & Tolias 2019; Hsu et al. 2018):

* **5 sequential tasks**, each containing 2 digit classes:
    - Task 1: digits 0, 1
    - Task 2: digits 2, 3
    - Task 3: digits 4, 5
    - Task 4: digits 6, 7
    - Task 5: digits 8, 9
* **No task IDs at training or test time** — the brain is never told which task it's on. This is the hardest variant; domain-incremental and task-incremental are easier and excluded from the headline.
* **Strict sequential training**: train Task 1 to convergence, freeze the experiment (no more Task 1 episodes), train Task 2 to convergence, etc. The brain only ever sees one task's data at a time.
* **Action space remains 10 digits throughout** — the brain must learn to never output a digit it hasn't seen yet, and to retain old digits as new ones are added.

### Why this should work architecturally

The reasoning that motivates running this experiment:

* Sensory neurons for digit-0 patterns and digit-2 patterns have **disjoint context fingerprints** — the spatial co-activation patterns across 784 channels are different, so they activate different inter-channel connections.
* When training Task 2, no Task 1 patterns fire (their spatial contexts don't match), so their action connections aren't modified.
* **Higher-level patterns live exponentially longer than lower-level ones.** The decay schedule is stratified by level: base sensory neurons decay fastest, level-1 patterns slower, level-2 patterns slower still, and so on. Whatever digit-0 patterns the brain consolidated up the hierarchy during Task 1 training have decay timescales that comfortably exceed the duration of Tasks 2–5.
* This means the relevant retention question is not "does the forget rate eat Task 1 patterns" but "did Task 1 train long enough to push patterns up to durable levels." If yes, retention is essentially free.
* Action connections on those high-level patterns are similarly persistent — they were established when the patterns were the strong predictors of the digit-0 reward, and nothing in Tasks 2–5 fires those patterns to modify them.

The risk shifts accordingly: it's not "forget rate too aggressive" but "Task 1 trained too briefly to consolidate to durable levels." This is a training-schedule question, not a decay-parameter question.

### Evaluation

After each task is trained, measure accuracy on the held-out test sets of **all tasks seen so far**. Report:

1. **Final retention matrix** (5×5): accuracy on Task i after training through Task j, for all i ≤ j. The diagonal is "just-trained" accuracy; the bottom row is "after all training" retention.
2. **Average accuracy after Task 5**: mean of the bottom row. This is the headline number.
3. **Forgetting metric**: for each task, max accuracy ever achieved minus final accuracy. Lower is better.

### Baselines (cited, not re-run)

Reference numbers from the continual learning literature for class-incremental Split-MNIST without replay or task IDs:

* **Naive sequential MLP/transformer fine-tuning**: ~20% average accuracy after Task 5 (van de Ven & Tolias 2019; Hsu et al. 2018). This is the catastrophic-forgetting floor.
* **EWC (Elastic Weight Consolidation)**: ~20–25%. Regularization-based methods barely help in class-incremental setting.
* **Replay-based methods (iCaRL, GEM)**: 70–90%, but these require storing old data and are explicitly outside the "no replay" claim.
* **Joint training upper bound**: ~98% (all tasks trained together — not a continual learning method).

We cite these numbers rather than re-running. Reproduction is a separate effort if external review demands it.

### Phase 2 success criteria

* **Headline**: average accuracy after Task 5 substantially above the ~20% catastrophic-forgetting floor, achieved with **no replay buffer, no task IDs, no regularizers** — purely from architectural retention.
* **Stretch**: retention competitive with replay-based methods (60%+) without using replay.
* **Diagnostic**: per-task accuracy degradation profile. If Task 1 accuracy is preserved through Task 5, the architectural claim holds; if it decays, forget-rate tuning is needed.

### Hyperparameter notes for Phase 2

The critical knob is **per-task training duration**, not forget rate. Each task must train long enough for digit patterns to consolidate up the hierarchy to levels with decay timescales exceeding the remaining experiment duration. Under-training a task leaves its representations at low levels that decay during subsequent tasks; over-training is harmless beyond saturation.

Recommended starting point:
* Match the per-task episode count to whatever achieved >95% within-task accuracy in Phase 1 (vanilla MNIST). If Phase 1 converged in 30 episodes, use 30+ episodes per task in Phase 2.
* Forget rate stays at the Phase 1 value (0.001–0.01). The level-stratified decay schedule does the work.
* If retention is weak, the first diagnostic is to inspect the level distribution of digit-0 patterns after Task 1 training. If they're concentrated at low levels, train Task 1 longer. If they're at high levels and still being lost, then forget rate is the issue.

### What this is NOT

Not an attempt to beat continual learning SOTA methods that use replay. The claim is structural: we get retention from the architecture, not from buffering or regularizing. A 60% result with no replay is a stronger paper than a 90% result with replay, because the former is a different category of system.

### Dependencies

* Phase 1 (vanilla MNIST) complete and working
* Same Rust core, same channel code — no architectural changes
* Sequential task scheduler (trivial — just feed task data in order)
* Per-task evaluation harness (trivial)

---

## 7. Calculate up/down accuracy separately

- Report directional accuracy (up vs down) independently to identify prediction bias

---

## 8. Neuron Limits

### Max neuron count hyperparameter
- Add configurable cap on neuron count per region/column

### Capacity enforcement
- When capacity is reached, stop learning new patterns
- Once forgetting frees space, resume learning automatically

### Overflow warning
- Emit warning when capacity is hit so the operator knows the brain is saturated
- Test that learning resumes correctly after decay opens capacity

---

## 9. Exponential Temporal Binning Test

Implement the cortical temporal binning scheme described in [experiment-temporal-binning.md](./experiment-temporal-binning.md).

### Summary
Higher-level patterns currently store context at exact frame distances — meaningless precision at their timescale. Exponential bins give every level the same number of bins but scale bin width with level, letting higher-level patterns represent long-range temporal relationships without explosion in context entries.

### Key changes
- Context struct stores bin index instead of exact distance
- Bin conversion: `distanceToBin(distance, level, contextLength, numBins)`
- Pattern matching uses bin-space comparison
- Voting carries bin index instead of exact distance

### Validation
- Level-1 patterns behave nearly identically to current (regression)
- Higher-level patterns form with fewer, coarser context entries
- Prediction accuracy on existing benchmarks does not regress

---

## 10. Documentation & Publish

### Update all documentation
- Sync docs with current architecture post-Rust migration
- Update README demos and examples

### npm package
- Prepare package for publication
- Publish to npm registry
