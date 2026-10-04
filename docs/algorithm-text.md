# A stream of letters

A worked case for [algorithm.md](algorithm.md): one channel of letters, one letter per frame, and the words
that form from them. Nothing here is normative. It follows the definitions from the base, where a letter sees
only the letter before it, to the level at which a word stands as one neuron.

---

# 1. The environment

One channel, `text`, laid out over time alone. It declares one event dimension, `letter`, with a bucket per
letter and one for the space. One frame is one letter. There is no action dimension: this machine reads.

# 2. The base: one letter back

At the base the reach is 1 (D4), so a letter's neighborhood is one neighbor, the letter before it. A row of one
neighbor carries two relations (D47): the owner with that one neighbor, and, across rows, the different letters
that have stood there.

- **A class of the letters before it.** Different letters stand before `h`, so `h`'s table holds a variable at
  one back, a class whose members are the letters that have recurred there (D41, D47). A bid covers its bidder
  (D31), so where the class holds, two letters are written as one call. Its value neuron for the letter that
  stood fires at `h`'s frame (D45): the one for `t` is the pair, and this case writes it `th`.
- **A neighbor where nothing varies.** Where only one letter has stood before another within its history, there
  is nothing to choose among, and the table holds a function that names that one neighbor. Once a class holds the
  position, no function names a letter there (D27).

# 3. Words, on the frontier

A child sees the frontier as far back as its reach: other children, and the letters the base left uncovered
(D5). Its reach is twice what it spans (D4).

Take `the`. `th` fires at `h`'s frame, and the bid that fired it covers `t` and `h`. `e` is uncovered. `e`'s own
neighborhood at reach 1 holds `th` one frame back, one of the things that stand before `e`, so it is a member of
`e`'s class there, and the class's value neuron for it is `the`, one level above the highest thing its bid
covered, at level 2 (D2). `the` spans two frames, so it sees four back, where the space before the word stands,
and `the` with a space before it is the word as a word.

A word of any length forms this way: a letter at a time, one level per letter, or faster where two children join,
`ca` and `ts` making `cats` in one step, `ca` a member of the class `ts` holds two back. Nothing has to be a power
of two long, because a stranded letter is never stranded; it is on the frontier, and whatever fires next within
reach can name it. And a long word does not see far for standing high: `strengths`, built a letter at a time,
stands eight levels up, spans eight frames and sees sixteen back.

# 4. A doubled letter

`ll`, `oo` and `ss` are first pairs like any other: the second `l` sees the first one back, a neuron being its
own neighbor at any offset but zero (D26), and `l` is a member of the class before `l`. A doubled letter as
such is caught a level up, in the table of a child that sees two frames back: where the two frames before it
hold one and the same letter in row after row, whatever the letter, that is a parameter (D41), found by its own
collapse (D27). Its value neuron carries which letter was doubled.

# 5. What the frontier holds after a sentence

Children for the words that recur, standing at their levels; value neurons for the pairs inside words the
machine has not yet seen whole; the letters nothing covered; and the
spaces, which are letters like any other until a pattern names them. The frontier is what the next letter sees,
what connects to whatever follows, and what would vote if the channel had anything to do (R27).
