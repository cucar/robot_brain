# A stream of letters

A worked case for [algorithm.md](algorithm.md): one channel of letters, one letter per frame, and the words
that form from them. Nothing here is normative. It follows the definitions from the base, where a letter sees
only the letter before it, to the level at which a word stands as one neuron.

---

# 1. The environment

One channel, `text`, laid out over time alone. It declares one event dimension, `letter`, with a bucket per
letter and one for the space. One frame is one letter. There is no action dimension: this machine reads.

# 2. The base: one letter back

At the base the reach is 1 (D4), so a letter's neighborhood is one cell, the letter before it. A row of one cell
carries two relations (D47): the owner with that one cell, and, across rows, the different letters that have
stood there.

- **Functions for the frequent pairs.** `h` has `t` one back in many of its rows. A bid covers its bidder (D31),
  so the function `(t, one back)` in `h`'s table writes two letters as one call, and its child `th` fires at
  `h`'s frame whenever the pair recurs. Every pair frequent enough to pay for its line gets one, and these are
  most of what forms at the base.
- **Classes for the rare pairs.** The letters that stand before `h` too seldom to earn a function each are held
  by a class at one back (D41): it pays by narrowing the letter to one of its members. Where it holds, its value
  neuron fires at `h`'s frame and says which letter it was (D45). A letter that grows frequent there takes its
  cell from the class for a function of its own, once the choice it stops writing pays for the line (D27, D30).

# 3. Words, on the frontier

A child sees the frontier as far back as its reach: other children, and the letters the base left uncovered
(D5). Its reach is twice what it spans (D4).

Take `the`. `th` is bought at `h`'s frame and covers `t` and `h`. `e` is uncovered. `e`'s own neighborhood at
reach 1 holds the `th` child one frame back, so `e`'s table forms the function `(th, one back)`, and its child
`the` stands one level above the highest thing it covered, at level 2 (D2). `the` spans two frames, so it sees
four back, where the space before the word stands, and `the` with a space before it is the word as a word.

A word of any length forms this way: a letter at a time, one level per letter, or faster where two children
join, `ca` and `ts` making `cats` in one step. Nothing has to be a power of two long, because a stranded
letter is never stranded; it is on the frontier, and whatever fires next within reach can name it. And a long
word does not see far for standing high: `strengths`, built a letter at a time, stands eight levels up, spans
eight frames and sees sixteen back.

# 4. A doubled letter

`ll`, `oo` and `ss` are first pairs like any other: the second `l` sees the first one back, a neuron being its
own neighbor at any offset but zero (D26), and `l`'s table holds `(l, one back)`. The doubles too rare for a
function of their own are caught a level up, in the table of a child that sees two frames back: where the two
frames before it hold one and the same letter in row after row, whatever the letter, that is a parameter (D41),
found by its own collapse (D27). Its value neuron carries which letter was doubled.

# 5. What the frontier holds after a sentence

Children for the words that recur, standing at their levels; children for the pairs inside words the machine
has not yet seen whole; value neurons for the rare letters a class held; the letters nothing covered; and the
spaces, which are letters like any other until a pattern names them. The frontier is what the next letter sees,
what connects to whatever follows, and what would vote if the channel had anything to do (R27).
