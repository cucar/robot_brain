# Variable-length pricing

[algorithm.md](algorithm.md) prices the file with a **fixed-length code**: one neuron is one symbol, and every
symbol costs the same whether it appears in every frame or in one. Every cost in the design is therefore a count
of neurons — a configuration's storage is how many neurons it names, and an error is how many neurons were got
wrong.

This document is the alternative: price the same file with a **variable-length code**, where a symbol costs
according to how often that neuron actually occurs. Nothing about the machine changes. The file is identical, the
sections are identical, and only the exchange from symbols to length differs.

## Forgetting is not a separate mechanism

A neuron keeps a [history](algorithm.md#the-history) one horizon wide and deletes a child the moment it stops
covering its own storage. Forgetting is the delete half of the [one test](algorithm.md#the-one-test) and needs no
machinery of its own, so what remains here is one question about how the file is priced.

## What changes if symbols are priced by occurrence

A neuron that fires in most frames is cheap to name; one that fires rarely is expensive. That flows into all
three costs at once:

- **Activating an apex neuron** stops costing 1. A pattern in constant use is nearly free to record; a rare one
  is not.
- **Errors** stop being a plain count. Getting a neuron wrong that is almost always in the same state costs more
  than getting a genuinely uncertain one wrong.
- **Having a child** costs according to which neurons its configuration names, not how many.

Because the one test compares stored cost against errors removed, and both sides move, every structural decision
shifts — a child naming common neurons becomes cheap to keep, and one naming rare neurons has to work harder.

## Why it is not the design

It is more machinery for an unproven gain. Fixed-length counting gives the same answer wherever occurrence is not
badly skewed, and it optimises the standing metric — apex neurons per level — directly and without a single
logarithm. The variable-length version only pays for itself if occurrence really is skewed enough to change
decisions, and that is a measurement, not an assumption.

It also introduces a dependency the fixed-length version does not have: every price becomes a function of
frequencies that are themselves still being learned, so early prices are unstable and every number moves at once
when the estimates move.

## Measuring it

Run both, identical otherwise, on the standing metrics: apex neurons per level per frame, the dictionary size
that bought them, and held-out accuracy. Fixed-length is the baseline; variable-length has to beat it.

The place to look for a difference is where the fixed-length code is deliberately blind — the match. Under
counting, distance is the number of neurons an entry gets wrong and every position weighs the same. Under
occurrence pricing, a mismatch in a position that is nearly always the same weighs more. If variable-length
pricing wins anywhere, it should show up as better-shaped configurations rather than as fewer of them.

Note that it also breaks a property the fixed-length code has for free: with equal weights the best configuration
for a set of frames is their coordinate-wise median, computed directly. Weighted distances make
[refinement](algorithm.md#refinement) a weighted median instead — still computable, but no longer the same
one-line rule.

## Open questions

- **Which frequencies.** Occurrence measured over what — all frames, frames where the parent was active, frames
  where the entry served? Each gives a different code and a different set of decisions.
- **Non-stationarity.** Occurrence counts never decay while everything measured against them does, so on drifting
  domains every price is set against an aging estimate. Windowing them perturbs every number at once.
