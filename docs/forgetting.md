# Forgetting

[algorithm.md](algorithm.md) prices the file with a **fixed-length code**: one neuron is one symbol, and every
symbol costs the same whether it appears in every frame or in one. Rent follows from that — a pattern's cost is
the length of its line in the dictionary, divided by the horizon.

This document is the alternative: price the same file with a **variable-length code**, where a symbol costs
according to how often that neuron actually occurs. Nothing about the machine changes. The file is identical, the
sections are identical, and only the exchange from symbols to length differs.

## What changes if symbols are priced by occurrence

A neuron that fires in most frames is cheap to name; one that fires rarely is expensive. That flows into all
three costs at once:

- **Activating an apex neuron** stops costing 1. A pattern in constant use is nearly free to record; a rare one
  is not.
- **Errors** stop being a plain count. Getting a neuron wrong that is almost always in the same state costs more
  than getting a genuinely uncertain one wrong.
- **Having a pattern** costs according to the neurons its line names, not how many.

And because rent is the pattern's cost over the horizon, rent itself becomes usage-dependent. **That is what
makes this dynamic forgetting**: a heavily used pattern is cheap to keep and survives on little, while a rarely
used one is expensive and is culled sooner. Under the fixed-length code every pattern of the same size pays the
same rent regardless of how central it is.

## Why it is not the design

It is more machinery for an unproven gain. Fixed-length counting gives the same answer wherever the alphabets are
similar and usage is not badly skewed, and it optimizes the standing metric — apex neurons per level — directly
and without a single logarithm. The variable-length version only pays for itself if occurrence really is skewed
enough to change decisions, and that is a measurement, not an assumption.

It also introduces a dependency the fixed-length version does not have: every price becomes a function of
frequencies that are themselves still being learned, so early prices are unstable and every number moves at once
when the estimates move.

## Measuring it

Run both, identical otherwise, on the standing metrics: apex neurons per level per frame, the dictionary size
that bought them, and held-out accuracy. Fixed-length is the baseline; variable-length has to beat it.

The place to look for a difference is where the fixed-length code is deliberately blind — the match. Under
counting, distance is the number of neurons an entry gets wrong, and every position weighs the same. Under
occurrence pricing, a mismatch in a position that is nearly always the same weighs more. If variable-length
pricing wins anywhere, it should show up as better-shaped configurations rather than as fewer of them.

## Open questions

- **Which frequencies.** Occurrence measured over what — all frames, frames where the parent was active, frames
  where the entry was matched? Each gives a different code and a different rent.
- **Non-stationarity.** Occurrence counts never decay while everything measured against them does, so on drifting
  domains every price is set against an aging estimate. Windowing them perturbs every number at once.
