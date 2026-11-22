# Shape Demos

Simple images on a grid of pixels: lines, squares, and a 2 and a 3. Each image is one channel over x and y with one
event dimension, a pixel black (0) or white (1); a second channel, `digit`, holds ten digit actions. For the other
demos see [mnist-demos.md](mnist-demos.md), [stock-demos.md](stock-demos.md), [text-demos.md](text-demos.md) and
[tap-demos.md](tap-demos.md).

Each showing is two frames: the image, then the digit with a reward of +1. The context is reset after each showing,
so no showing sees another. Every line prints what the brain said, the levels, what stands on the apex, the functions,
and whether the dictionary changed; after the last showing, what stands on the apex and the relations among it; at
the end, the dictionary.

```bash
node apps/shapes/jobs/shapes.js <hline|vline|square|hollow|teach23> [showings=30]
```

---

## Lines and squares

One image shown 30 times to one brain, each showing taught the digit 0.

```bash
node apps/shapes/jobs/shapes.js hline 30
node apps/shapes/jobs/shapes.js vline 30
node apps/shapes/jobs/shapes.js square 30
node apps/shapes/jobs/shapes.js hollow 30
```

| image | said 0 | functions | the dictionary last changed | on the apex at the end |
|---|---|---|---|---|
| a white horizontal line of 5 pixels in the middle row of a black 7×5 image | 29 of 30 showings | 7 | showing 7 | 1 call |
| a white vertical line of 5 pixels in the middle column of a black 5×7 image | 29 of 30 showings | 7 | showing 7 | 1 call |
| a 5×5 image, a black border around a white 3×3 middle | 29 of 30 showings | 7 | showing 6 | 1 call |
| a 5×5 black border whose 3×3 middle reports nothing | 29 of 30 showings | 5 | showing 5 | 1 call |

The first showing says nothing: the brain has not yet been taught a digit. From the second on it says 0 every time,
and the dictionary stops changing within the first seven showings; at the end one call stands on the apex.

**Expected output** (the horizontal line, the end of its showings):
```
  hline showing 30: said 0 | levels 5 | on the apex 1 | functions 7 | dictionary same
  on the apex: 1 calls, 0 bare pixels
  hline: the dictionary last changed at showing 7 (unchanged over the last 23 showings)
```

---

## The 2 and the 3

The images of a 2 and a 3, each 7×9, taught in turn with their digit for 30 rounds. Then each is shown without its
digit to a copy of the brain loaded with learning off.

```bash
node apps/shapes/jobs/shapes.js teach23 30
```

| | |
|---|---|
| the 2 said 2 | 27 of 30 rounds |
| the 3 said 3 | 27 of 30 rounds |
| with learning off, the 2 without its digit | says 2 |
| with learning off, the 3 without its digit | says 3 |
| functions at the end | 23 |
| edges at the end | 42 |
| the dictionary last changed | round 29 |

Each of these runs in under a second. They are the `hline`, `vline`, `filled-square`, `hollow-square` and
`two-and-three` benchmarks; their full outputs are saved in [benchmarks/output](benchmarks/output) (see
[Benchmarks](../README.md#benchmarks)).
