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

- **Bigrams.** `h` has `t` one back in most of its rows, so `h`'s table holds the pattern `(t, one back)`, and
  its child `th` fires at `h`'s frame whenever the pair recurs. Every frequent pair gets one.
- **Classes of letters.** At one back of `h`, `t`, `s`, `c` and `w` have each stood; where naming the set saves
  more than naming any one of them, `h`'s class table holds that class, and a pattern of `h` with the class
  covers `th`, `sh`, `ch` and `wh` at once (D41). Beside its child a value neuron says which (D45).

# 3. Words, on the frontier

A level-1 child has a reach of 2 and sees the frontier two frames back: other children, and the letters the
base left uncovered (D5).

Take `the`. `th` is bought at `h`'s frame and covers `t` and `h`. `e` is uncovered. `e`'s own neighborhood at
reach 1 holds the `th` child one frame back, so `e`'s table forms the pattern `(th, one back)`, and its child
`the` stands one level above the highest thing it covered, at level 2 (D2). `the` then sees four frames back,
where the space before the word stands, and `the` with a space before it is the word as a word.

A word of any length forms this way: a letter at a time, one level per letter, or faster where two children
join, `ca` and `ts` making `cats` in one step. Nothing has to be a power of two long, because a stranded
letter is never stranded; it is on the frontier, and whatever fires next within reach can name it.

# 4. A doubled letter

`ll`, `oo` and `ss` are one neuron at two adjacent offsets. In the table of the letter that follows them, the two
offsets hold one neuron in row after row, whatever letter it is: a parameter (D44), found by its own collapse
(D27), and named by the patterns of that letter where it recurs. Its value neuron carries which letter was
doubled.

# 5. What the frontier holds after a sentence

Children for the words that recur, standing at their levels; bigram children for the pairs inside words the
machine has not yet seen whole; the letters nothing covered; and the spaces, which are letters like any other
until a pattern names them. The frontier is what the next letter sees, what connects to whatever follows, and
what would vote if the channel had anything to do (R27).
