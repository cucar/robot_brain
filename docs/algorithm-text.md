# A stream of letters

A worked case for [algorithm.md](algorithm.md): one channel of letters, one letter per frame, and the words
that form from them. Nothing here is normative. It follows the definitions from the base, where a letter sees
only the letter before it, to the level at which a word stands as one neuron.

---

# 1. The environment

One channel, `text`, laid out over time alone, with radius `R = 1` (D1): a letter sees the letter before it
and nothing further. It declares one event dimension, `letter`, with a bucket per letter and one for the space.
One frame is one letter. There is no action dimension: this machine reads.

# 2. The base: one letter back

At the base the reach is the radius, here 1 (D4), so a letter's neighborhood is one neighbor, the letter before
it. A row of one neighbor carries two relations (D47): the letter and the one before it standing together, and,
across rows, the different letters that have stood one back from it.

- **A class over the letters before it.** Different letters stand before `h`, so `h`'s table holds a
  function naming `h` and a class at `−1` from it, whose members are the letters that have recurred there
  (D38, D41, D47). Where a member stands, two letters are priced as one call: the function's child fires at
  `h`'s frame, its owner's (D53), and the letter that stood stays on the apex one frame back, bound (D10). This
  case writes the child `·h`, a letter and then `h`.
- **A neighbor where nothing varies.** Where only one letter has stood before another within its history, there
  is nothing to choose among, and the table holds a function that names the letter and that one neighbor. Once
  a class holds the coordinate, no function names a letter there (D27).

# 3. Words, on the frontier

A child sees the frontier as far back as its reach: other children, and the letters the base left uncovered
(D5). Its reach is what it spans plus the radius, one (D4).

Take `the`. `·h` fires at `h`'s frame, covering `h`, and `t` stands bound a frame before it. `e` is
uncovered. `e`'s own neighborhood at reach 1 holds `·h` one frame back, one of the things that stand before
`e`, so it is a member of `e`'s class there, and the call of `e`'s function with that member is `·he`, a
frame above `·h`. `·he` spans two frames, so it sees three back: the bound `t`, and the space before the word.
A function in `·he`'s table naming it, `t` two back and the space three back is `the` as a word, one neuron; a
bound letter is a neighbor like any other (D7, D10).

A word of any length forms this way: a letter at a time, one level per letter, or faster where two children join,
`ca` and `ts` making `cats` at once, `ca` a member of the class the function in `ts`'s table names two back.
Nothing has to be a power of two long, because a stranded letter is never stranded; it is on the frontier, and
whatever fires next within reach can name it. And a long word does not see far for standing high: `strengths`,
built a letter at a time, stands eight levels up, spans eight frames and sees nine back. A word is written in the
offsets between its own letters (D15), so a child whose window holds a word it has seen before names it with one
line wherever in the window it stands, and two children that see one word bid one neuron for it (R43).

The radius sets how much of this happens at the base. At `R = 4` a letter sees four letters back, each at its
own offset (D6), so a word of up to five letters is one function in its last letter's table and one level up,
and the chain above starts from there; [algorithm-xy.md](algorithm-xy.md) is a stream declared that way.

# 4. A doubled letter

`ll`, `oo` and `ss` are first pairs like any other: the second `l` sees the first one back, a neuron being its
own neighbor at any offset but zero (D26). The owner and `l` at `−1` is a seed in `l`'s table (D47), and once
the pair has come round twice the collapse makes it a function, `ll`, with a child (D33, D27). The same
happens for `oo` in `o`'s table and `ss` in `s`'s. A doubled letter that comes often is covered by its own
letter's function, once per kind of doubling, and each occurrence pays the owner's name and which pattern.

A parameter covers all doublings with one function, but every occurrence then pays which member besides
(D13). For a dozen kinds that each recur, the functions are cheaper, so that is what the pick builds first, and
their children cover the doubled letters before anything above could see them. What a parameter catches
is the doubling too rare to have earned a function: a level up, in the table of a child that sees two frames
back, two adjacent frames holding one and the same letter in use after use is a same-to-same relation, and the
function grown from it ties the two places in a parameter (D44); the doubled letter stays bound at both.

# 5. What the frontier holds after a sentence

Children for the words that recur, standing at their levels; the letters bound in their variables; the letters
nothing covered; and the
spaces, which are letters like any other until a pattern names them. The frontier is what the next letter sees,
what connects to whatever follows, and what would vote if the channel had anything to do (R27).
