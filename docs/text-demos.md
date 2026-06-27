# Text & Language Demos

Applying the brain to character-sequence prediction. For the vision and time-series demos see
[mnist-demos.md](mnist-demos.md) and [stock-demos.md](stock-demos.md).

---

## Text Sequence Learning

The brain learns to predict character sequences. Feed it a string, and it memorizes the pattern —
reaching ~99.96% prediction accuracy within two episodes and staying flat there.

Run the text test with hyperparameters tuned for text (the defaults are tuned for stock data):

```bash
node apps/text/jobs/test.js --file abramov.txt --group-mode static --group-threshold 0.9 --context-length 20 --forget-rate 0.00003 --no-summary
```

**Expected output:**
```
📊 Accuracy by Episode:
   Episode 1: 27.81% (32674 frames)
   Episode 2: 99.96% (32674 frames)
   Episode 3: 99.96% (32674 frames)
   Episode 4: 99.96% (32674 frames)
   Episode 5: 99.96% (32674 frames)
```

The brain goes from low accuracy to ~99.96% in two episodes and holds there — it has fully memorized the
character sequence except for the first ~20 characters at the start of each episode. Those leading characters
can't be predicted because the brain hasn't seen any context yet — it needs a `context-length` window of prior
characters in memory before it can recognize patterns and cast votes. The "warmup" frames at the head of each
episode are a structural property of context-based prediction, not a learning failure: every character past the
warmup window is predicted correctly.
