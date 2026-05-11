# Scale Stock Processing

## Goal

With the multi-threaded Rust core complete (Rayon column parallelism, R=1 C>1), focus on scaling the stock trading workload as the primary benchmark.

---

## Behavior baseline

**Baseline = the four documented demos in [README](../README.md).** They cover synthetic cycles, real stock data, sequence memorization, and text learning — together they exercise pattern creation, error correction, deletion cascades, and convergence dynamics.

| # | Demo | Command | Headline numbers to verify |
|---|------|---------|----------------------------|
| 1 | [Demo 1 — Single-channel synthetic cycle](../README.md#demo-1-single-channel-synthetic-cycle) | `node apps/stocks/jobs/synthetic-extended-test.js --error-mode static --error-threshold 0.3 --merge-threshold 0.9` | Overall Optimal Rate (e.g. `233/240 = 97.1%`) |
| 2 | [Demo 3 — Stock trading, 1 episode](../README.md#demo-3-stock-trading) | `node apps/stocks/jobs/test.js` | Episode 1 net profit, total trades, base-level accuracy |
| 3 | [Demo 6 — Sequence memorization, **1 episode only**](../README.md#demo-6-stock-sequence-memorization) | `node apps/stocks/jobs/test.js --no-summary --episodes 1 --symbols KGC,GLD,SPY --context-length 3 --forget-rate 0.001 --error-mode static --error-threshold 0.3` | Episode 1 net profit, total trades, base-level accuracy |
| 4 | [Demo 7 — Text sequence learning](../README.md#demo-7-text-sequence-learning) | `node apps/text/jobs/test.js --file abramov.txt --error-mode static --error-threshold 0.3 --context-length 20 --merge-threshold 0.9 --forget-rate 0.001 --no-summary` | Per-episode accuracy across all 5 episodes |

---

## 1. Multi-stock parallel processing
- Multiple stock channels processed in parallel across columns
- Benchmark: throughput vs single-threaded baseline

## 2. Performance tuning
- Inverted index for pattern recognition already implemented — validate selectivity metrics at scale
- Profile hot paths (vote aggregation, connection updates, index maintenance)
- Optimize memory layout for cache locality (arena allocation, struct-of-arrays where beneficial)
- Tune thread count and neuron partitioning for stock workloads

## 3. Add pre-training demo
- Change offset rows to work from the end and add demo for training first and then testing accuracy with hold out data

---

See [future-work.md](future-work.md) for MPI distribution, Python bindings, text/vision/audio channels, robotics, and other longer-term plans.
