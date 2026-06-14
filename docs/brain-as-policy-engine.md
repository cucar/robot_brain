# Brain as Policy Engine — Removing the Consensus

## Motivation

The brain currently does two unrelated things in one place:

1. **Maintain a learned model** of state → action value (connections, rewards, patterns).
2. **Pick winners** via a per-dimension consensus that forces "one bucket wins per dim".

(2) is application logic masquerading as brain logic. It encodes a specific decision policy — argmax over a normalized probability per channel-dim — that doesn't fit most domains:

- **Stocks.** "Both buy and sell" is a real prediction with semantic meaning ("I'm uncertain"). Forcing a winner destroys the information the trading layer would actually use.
- **Text generation.** Sampling by probability requires the *distribution*, not the argmax. Top-k, top-p, temperature — all live in the app.
- **MNIST.** Argmax over digit actions is the right *shape*, but it's a one-liner that doesn't need brain-side machinery — and, as the empirical note below shows, the brain's *particular* argmax (a weighted mean) is not even the best rule: a Naive-Bayes product over the same votes beats it by ~5pp.
- **Weather ("will it rain?")**. The honest answer is a probability — and it doesn't need to sum to 1 across alternatives, because the alternatives aren't mutually exclusive ("rain *and* hail" is allowed).

The brain should produce **action votes with strengths and rewards**. Apps decide what to do with them.

## Empirical validation: the app-side rule already beats the brain's consensus (MNIST)

This is not hypothetical. On MNIST, an app-side aggregation of the kind this refactor moves out *already outperforms* the brain's built-in consensus — which is the strongest possible argument that "pick winners" is misplaced application logic.

The brain's consensus picks the digit with the highest **strength-weighted arithmetic mean** of the per-voter posteriors `P(d|voter)` (`aggregate_votes` → `determine_dimension_winners`: an action's score is `weighted_total / strength`, then argmax). Reaggregating the *same* votes in the MNIST job with a **Naive-Bayes product** rule instead — `argmax_d Σ_voter log(P(d|voter) + ε)` — gives a consistent **+5pp test accuracy**, with no retraining and no brain change:

| config (binary, 1000-image test) | brain consensus | NB product | lift |
| --- | --- | --- | --- |
| 14×14, 300/class | 84.3% | 89.5% | +5.2pp |
| 28×28, 500/class | 83.8% | 89.3% | +5.5pp |

Why the product wins: the mean is a soft ensemble — a digit can win on a mediocre average even when several voters strongly contradict it — whereas the product respects each voter's **veto** (`P(d|v) ≈ 0 → log ≈ −large`), which is the correct combination rule for an argmax over mutually-exclusive classes with roughly-independent evidence. The brain's consensus is built for expected-reward action *selection*, not classification, so it systematically under-reads the evidence the hierarchy already learned.

The point for this refactor: even for MNIST — the one app where "argmax" is the right shape — the brain's *particular* argmax is not the best rule. The decision policy genuinely belongs in the app. This is implemented today behind `--decode nb` in `apps/mnist/jobs/test.js`, which reads the votes the brain already exposes via `setEmitVotes(true)` and computes the log-sum in ~10 lines of JS — a working preview of the votes-only consumption this design makes the default.

## The new API

### `Brain::process_frame`

```rust
pub fn process_frame(
    &mut self,
    events: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>,
    actions: &FxHashMap<ChannelId, FxHashMap<DimensionId, Vec<(f64, Reward)>>>,
) -> FrameResult
```

- **`events`**: sensory observations as scalar values. Same shape as today's `inputs`.
- **`actions`**: actions the app executed since last call, expressed as `(value, reward)` pairs.
  - One channel/dim can carry multiple executed actions (multi-asset trades, multi-character writes).
  - Brain resolves `value → bucket_id → neuron_id` internally using the same path `Brain::learn` already uses (`resolve_action_targets`).
  - Empty for the first frame, or for eval-only frames where the app doesn't want to learn from any action.
- **Returns** `FrameResult { votes: Vec<FrameVote>, elapsed, timings }`.

The current `rewards: &FxHashMap<ChannelId, Reward>` parameter and the `self.rewards: Vec<...>` history Vec on Brain are **deleted**. Rewards travel with their action; there's no need for a per-channel sliding window.

### `FrameVote`

```rust
pub struct FrameVote {
    pub voter_id: NeuronId,
    pub voter_label: String,
    pub voter_level: Level,
    pub target_id: NeuronId,
    pub channel_id: ChannelId,
    pub dim_id: DimensionId,
    pub value: f64,         // dequantized bucket value
    pub distance: Distance,
    pub strength: f64,
    pub reward: f64,
}
```

`target_type` is dropped — it's always `Action` under this model.

`value` is the dequantized bucket scalar so apps consume action votes the same way they produced action inputs: as values. ID resolution is brain-internal.

### `Brain::learn` — keep as-is, for supervised wiring

Single-frame supervised training (MNIST) doesn't fit the carry-forward model — the apex voter set must be wired to all candidate actions *immediately*, in the same frame the image was observed. `Brain::learn` already does this. Under the new architecture it stays:

```rust
pub fn learn(
    &mut self,
    actions: &FxHashMap<ChannelId, FxHashMap<DimensionId, Vec<(f64, Reward)>>>,
    distance: Distance,
) -> FrameResult
```

`Brain::learn` is for **supervised single-frame teaching**. `process_frame` is for **interactive / observational** learning.

### `Brain::infer` — gone

Read-only "what do you predict" used to mean "run consensus without learning". Under the new model, apps that want a no-side-effect read can call `process_frame(events, &empty_actions)` with `set_learning(false)` and read the votes. No separate API needed.

### `set_learning(false)` — kept

Eval mode still exists. It suppresses connection updates, error pattern minting, and Welford stats. Apps still need it for held-out evaluation, where they want the brain to predict without learning from the inputs.

## What disappears from the brain

**Code removed:**
- `aggregate_votes`, `determine_dimension_winners`, `determine_consensus`, `build_winners`, `build_inferences_by_channel`, `build_dim_inferences`, `infer_neurons`, `compute_inferences` — the consensus pipeline.
- `FrameResult.inferences`, `DimInferenceOutput`, `DimInference`, `Candidate`, `DimBestEntry` — consensus output shapes.
- `memory.inferred_neurons`, `save_inferred_neurons`, `get_inferred_neurons`, `clear_inferred_neurons` — winner persistence (nothing carries forward except executed actions, which arrive via API).
- `track_continuous_error` (MAPE) — app-layer concern, computed from votes if the app cares.
- `track_inference_performance`, `Diagnostics::track_inference_performance`, the accuracy/reward stat counters, `mispredictions` log — app-layer concerns.
- `self.rewards` Vec on Brain, `push_rewards`, the channel-keyed reward dispatch through `dispatch_temporal_frame`. Rewards become per-action.
- `Brain::infer` (the read-only sweep).

**Code simplified:**
- `learn_temporal_connections` filters actives to action targets only. Voters never learn connections to events or patterns; consensus-time "vote has no coordinate" panic root cause is structurally eliminated.
- `dispatch_temporal_frame` no longer takes `current_rewards: Option<&FxHashMap<ChannelId, Reward>>`. Per-action rewards arrive directly on `ActiveNeuron` from the API-side actions list.
- `get_frame_neurons` no longer reads `memory.inferred_neurons` for carry-forward; actions come straight from the API.
- The frame-summary `accuracy_correct`, `accuracy_total`, `avg_reward`, `mape`, `mispredictions` fields go away. Apps surface their own accuracy in the renderer tail.

**Approximate code reduction:** roughly 400-600 lines of brain core code removed; ~6 fewer types; the dispatch chain shrinks because rewards aren't threaded through it.

## Frame summary line, after

```
Frame N | Neurons: K (T{maxTemporalLevel} S{maxSpatialLevel}) | {app-specific tail} | Time: …ms
```

Accuracy / reward / MAPE move into the tail and are job-supplied. The brain-owned section becomes just structural state.

## Application migration

### MNIST

**Today:**
```js
brain.resetContext();
brain.processFrame(encoder.encodeImage(image), EMPTY_REWARDS);
brain.learn(encoder.encodeAction(label), 1);
// for eval:
const result = brain.processFrame(image, EMPTY_REWARDS);
const inferences = result.inferences;  // brain's argmax
const predicted = decodeDigit(inferences);
```

**After:**
```js
brain.resetContext();
brain.processFrame(encoder.encodeEvents(image), EMPTY_ACTIONS);  // events only, no executed actions
brain.learn(encoder.encodeAllLabels(label), 1);
// for eval:
brain.setLearning(false);
const result = brain.processFrame(image, EMPTY_ACTIONS);
const predicted = pickNaiveBayes(result.votes, ACTION_CHANNEL_ID);  // log-sum of posteriors; +5pp over the brain's weighted-mean argmax
```

`pickNaiveBayes` is ~10 lines of JS in the MNIST job: for each candidate digit, sum `log(vote.reward + ε)` over its action votes and take the argmax. The simpler `pickArgmaxByReward` (replicating the old brain consensus — `weighted_total / strength` per digit, argmax) is a valid fallback, but the NB rule is the recommended default per the empirical note above. Either way, the brain doesn't pick.

### Stocks

**Today:**
```js
const result = brain.processFrame(quotes, rewardsByChannel);
const inferences = result.inferences;
const trades = decideTradesFromInferences(inferences, portfolio);
executeTrades(trades);
// portfolio adjusts; rewards come next frame in rewardsByChannel
```

**After:**
```js
const result = brain.processFrame(quotes, executedActionsLastFrame);
const candidateTrades = decideFromVotes(result.votes, portfolio);  // app threshold + position sizing
const executedTrades = executeTrades(candidateTrades);
const executedActions = executedTrades.map(t => ({ channel: t.symbol, dim: 'side', value: t.side, reward: t.realizedPnL }));
// pass executedActions on next call
```

The app's consensus logic moves from "translate inferences into trades" to "compute scores from votes and decide trades". Same logic, different input shape (votes instead of inferences). The app already does most of this.

### Text

The text app currently uses event-based prediction (predict next character via consensus over the character channel). The migration:

- Reframe character prediction as an **action**: the brain *acts* by emitting a character.
- App receives action votes for the character channel.
- App samples or argmaxes according to its own decoding strategy (greedy, top-k, top-p, temperature).
- Selected character becomes the executed action passed back next frame.

This is the bigger app migration. Existing event-based logic gets replaced. Worth its own sub-plan.

## Resolved design points

- **No default consensus helper.** Apps know their own decision rule; the brain shouldn't ship a particular argmax.
- **No legacy snapshot migration.** Old backups are not preserved; restore discards any consensus-era state.
- **Action carry-forward must be the *executed* action.** Apps that distinguish intended-vs-filled (e.g., partial trade fills) must pass the latter. The brain learns from actual outcomes.
- **Distance on `Brain::learn`.** Keep `distance` as a parameter; supervised wiring is fundamentally a temporal-prediction concept. `distance=0` is not on the supervised path.

## Text app reframing

The text app is the largest application-side change. Today it's pure event prediction:
- Single channel `text` with one INPUT dim `text_char` (resolution 256, passthrough)
- Each frame: one character as an event keyed by ASCII code
- No actions, no rewards, `emitsReward: false`, `learnActionSequences: false`
- Brain predicts next character via per-dim consensus on `text_char`
- Accuracy = was the predicted next character equal to the actual next character?

Under the policy-engine model, text generation **is action**: the brain emits a character. The reframing:

### Channel spec change

```js
getChannelSpec() {
    return {
        name: this.name,
        emitsReward: false,
        learnActionSequences: true,
        dimensions: [
            {
                name: 'text_char_in',   // what was just read from the text stream
                kind: 'input',
                resolution: 256,
                mode: 'passthrough',
            },
            {
                name: 'text_char_out',  // what character the brain would emit next
                kind: 'action',
                resolution: 256,
                mode: 'passthrough',
            },
        ],
    };
}
```

Two dims: one input (observed character), one action (predicted next character). Same bucket space (256 ASCII codes), same passthrough mode.

### Training flow (supervised)

Mirrors MNIST: each frame is `processFrame(events, []) + learn(allActions, 1)`. The encoder needs a one-character look-ahead.

```js
async runFrame() {
    const cur = encoder.nextFrame();
    if (!cur) return false;
    const next = encoder.peek();        // peek without advancing — new encoder method
    if (!next) return false;            // end of stream: skip the last char (no target)

    // Observation: the character that just arrived.
    const events = new Map();
    events.set(encoder.channelId, new Map([[encoder.charDimInId, cur.charCode]]));

    // Supervised target: next char gets reward=1, every other char gets reward=0.
    const actionTargets = new Map();
    const charBuckets = new Map();
    for (let code = 0; code < 256; code++) {
        const reward = (code === next.charCode) ? 1 : 0;
        charBuckets.set(code, reward);
    }
    actionTargets.set(encoder.charDimOutId, charBuckets);

    brain.processFrame(events, EMPTY_ACTIONS);
    brain.learn(actionTargets, 1);   // wire apex voters → all char actions at d=1
    return true;
}
```

Per-character `learn` with all 256 chars is heavier than MNIST's 10-class call, but the wiring path is the same and the cost is `O(apex_size × 256)` per frame. For typical apex sizes (a handful), that's negligible.

### Eval flow (no learning, generative)

```js
brain.setLearning(false);
encoder.resetFrames();
let lastPicked = null;          // executed action from last frame
const correct = { hits: 0, total: 0 };

while (true) {
    const cur = encoder.nextFrame();
    if (!cur) break;

    // Build the actions payload from last frame's pick. Reward = was it right?
    const actionsLastFrame = new Map();
    if (lastPicked !== null) {
        const reward = (lastPicked === cur.charCode) ? 1 : 0;
        actionsLastFrame.set(encoder.channelId,
            new Map([[encoder.charDimOutId, [[lastPicked, reward]]]]));
        correct.hits += reward;
        correct.total += 1;
    }

    const events = new Map();
    events.set(encoder.channelId, new Map([[encoder.charDimInId, cur.charCode]]));

    const result = brain.processFrame(events, actionsLastFrame);

    // App-side decoding: pick the next character from the action votes.
    lastPicked = pickGreedy(result.votes, encoder.channelId, encoder.charDimOutId);
    // Could also be: pickTopK, pickSoftmaxSampled, pickWithTemperature(...).
}

const accuracy = correct.total > 0 ? correct.hits / correct.total : 0;
```

`pickGreedy` is the app's local consensus — for each candidate character, sum `vote.strength × vote.reward` and take the argmax. ~10 lines of JS in the text job.

### Accuracy measurement, app-layer

The brain no longer tracks `accuracy_correct / accuracy_total / mispredictions`. The text job tracks them itself:

```js
this.episodeMetrics.accuracy = correct.hits / correct.total;
this.episodeMetrics.mispredictions = ...;  // app tracks if it cares about per-char confusions
```

The frame summary tail can include `Pred: 87.3% ('o'→'a' x12, ...)` — domain-specific, useful to the text job, irrelevant to the brain.

### Generative / sampling (optional follow-on)

Once the supervised loop is working, generative text becomes trivial: replace `pickGreedy` with `pickSoftmaxSampled(votes, temperature)`. The brain doesn't need to know.

```js
function pickSoftmaxSampled(votes, channelId, dimId, temperature = 1.0) {
    const scores = new Map();          // bucket -> exp(score/T)
    for (const v of votes) {
        if (v.channelId !== channelId || v.dimId !== dimId) continue;
        const cur = scores.get(v.value) ?? 0;
        scores.set(v.value, cur + v.strength * v.reward);   // accumulate raw
    }
    const weighted = [...scores].map(([b, s]) => [b, Math.exp(s / temperature)]);
    const total = weighted.reduce((a, [, w]) => a + w, 0);
    const pick = Math.random() * total;
    let acc = 0;
    for (const [b, w] of weighted) { acc += w; if (acc >= pick) return b; }
    return weighted[weighted.length - 1][0];
}
```

### Encoder changes

- Add `peek()` returning the next frame without advancing the cursor.
- Add `charDimInId` and `charDimOutId` to replace the single `charDimId`. `bindIds()` resolves both.

### Mispredictions

Today's "predicted X but got Y" is stored on the brain's diagnostics. It moves to the app. The text job computes it from its own `lastPicked`-vs-actual loop and emits it in the per-episode log.

### What text gains from this

- Temperature / top-k / top-p sampling becomes trivial — already free in the votes payload.
- The "is the brain confident?" question is answerable (high vote concentration on one char) rather than hidden behind argmax.
- Misprediction analysis stays in the text app where it belongs — the brain doesn't have a "this is a text-specific log".

## Migration plan — transitional

The work lands in stages, each leaving everything compiling and testing. Stage 1 keeps the old consensus output alive in parallel so each app migrates independently without breaking other apps.

### Stage 1 — brain core: new API plus old output

- Change `process_frame` signature to take `(events, actions)` with per-action rewards.
- Remove `self.rewards` Vec, `push_rewards`, channel-keyed reward dispatch.
- Filter `learn_temporal_connections` to action targets only (kills the "vote has no coordinate" panic at the source).
- Brain still computes consensus internally and returns `FrameResult.inferences` alongside the new votes payload. Apps that haven't migrated keep reading `inferences`; apps that have read `votes`.
- New per-action reward shape arrives via `actions`; old per-channel reward map is gone (apps that fed `rewards: {channel → reward}` need to convert at the call site even before they migrate to votes — small touch).

### Stage 2 — migrate apps

- **MNIST jobs**: rewrite to `processFrame(events, [])` + `learn(labels, 1)` + `pickArgmaxByReward(votes)` in eval. ~5 jobs.
- **Stocks jobs**: each of the 3-4 jobs reads votes, decides trades via its own threshold/sizing logic, passes back executed trades with rewards. The decision logic is already in the jobs; only the input shape changes.
- **Text app**: per the reframing section above — add the action dim to the encoder, switch training to `processFrame + learn`, switch eval to votes + app-side pick.

### Stage 3 — rip the brain-side consensus

Once all apps no longer read `inferences`:

- Delete `aggregate_votes`, `determine_dimension_winners`, `determine_consensus`, `build_winners`, `build_inferences_by_channel`, `build_dim_inferences`, `infer_neurons`, `compute_inferences`, `Brain::infer`.
- Delete `FrameResult.inferences`, `DimInferenceOutput`, `DimInference`, `Candidate`, `DimBestEntry`.
- Delete `memory.inferred_neurons`, `save_inferred_neurons`, `get_inferred_neurons`, `clear_inferred_neurons`.
- Delete `track_continuous_error`, `Diagnostics::track_inference_performance`, `track_continuous_error` field on FrameTimings, `mape*` counters, `mispredictions` log, `accuracy_correct/accuracy_total/avg_reward` stats.
- Snapshot format drops the `inferred_neurons` section (no migration needed — no old backups preserved).

### Stage 4 — diagnostic cleanup

- Frame summary line: brain-owned section becomes `Neurons: K (T{maxTemporal} S{maxSpatial})` plus timing. Accuracy / reward / MAPE move to the job-supplied tail.
- `episode_summary` accuracy / mispredictions / reward stats: removed.

## Scope estimate

- **Brain core (Rust):** ~400-600 LOC removed, ~50 LOC added (signature changes), ~150 LOC refactored (learn_temporal_connections filter, dispatch_temporal_frame reward handling).
- **napi (Rust):** `processFrame` signature change, `getFrameSummary` cleanup, drop `inferences` from frame result, drop `track_continuous_error`. ~100 LOC.
- **Stocks app (JS):** 3-4 jobs rewritten. Each one's decision logic stays similar; the input shape changes. ~200-400 LOC across files.
- **MNIST app (JS):** ~5 jobs touched, each minor. ~50-100 LOC across files.
- **Text app (JS):** larger rewrite — action-based reframing of character prediction. Hard to estimate without inspecting current state.
- **Tests:** brain-core tests that exercise consensus are deleted (~15 tests). Replacement tests for the votes-only flow added.

Realistic total: **2-3 focused days** for stages 1-3 on the brain side, plus app-side migration time per app.

## What to lose sleep over

- The brain getting *worse* at any of the metrics that consensus used to dominate. If the app's consensus differs from the old brain consensus, accuracy numbers will move. That's expected — they should — but worth a baseline pre/post on a known-good run (MNIST 7×7 binary, say) to confirm nothing regressed beyond the metric-redefinition shift.
- Memory growth from rewards living on each connection rather than on Brain. (Should be a wash; each connection already carries a reward field.)
- Action carry-forward edge cases on the first frame and around `resetContext`. Empty actions list, no learning happens, vote results are uninformative — confirmable in tests.
