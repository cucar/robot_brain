# A stream of letters

A worked case for [algorithm.md](algorithm.md): one channel of letters, one letter per frame, and the stretches
that come to be named from them. Nothing here is normative. It follows the definitions from the base, where a
letter sees only the letter before it, to the interneurons that stand for the stretches that recur.

---

# 1. The environment

One channel, `text`, laid out over time alone, with radius `R = 1` (R4): a letter sees the letter before it
and nothing further (D4). It declares one event dimension, `letter`, with a bucket per letter and one for the
space; each bucket is a base neuron (D52). One frame is one letter. There is no action dimension: this machine
reads.

There is no spatial dimension, so a bare record is two numbers, the neuron and the frame (D12, D13). A call's
record is two numbers, plus one per relation that did not come true and one per value.

# 2. The base: one letter back

At the base the reach is the radius, 1 (D4), so a letter's neighborhood is one neighbor, the letter before it
(D7). Its activate call returns one relation per activation: the letter before, at `−1` in time, with the rows
of the history that hold it (D34, R21).

A relation held in one row is returned and read in the index, to find a function that already fits it; nothing
is built on it (R21, R53). Held in two rows, it is grouped into a function of two fixed nodes (R50). Take `th`.
Once `h`'s history holds `t@−1` in two rows, `{h, t@−1}` is a function: `t` and `h` one frame apart, written
from the older of the two (D15). Its price:

| | numbers |
|---|---|
| A, the two bare records | 4 |
| B, the call's record | 2 |
| savings per row | 2 |
| entry in the dictionary: one relation, two neurons | 3 |
| margin at two rows | 1 |

The margin is positive at two rows, so the function is kept (D30). Find reads its relation in the index and it
is new (R53). The election places it over `t` at frame `f − 1` and `h` at `f`, 4 for 2, and takes it (R24). Its
interneuron, call it `TH`, is allocated (R16) and fires at frame `f`, the latest frame of the two (D53),
covering both. `t`, open at age 1, connects once more, to `TH` at `+1` (D10); `h`, covered at age 0, writes
nothing. Both leave the apex (D49).

Once a function is in the dictionary, it is found from a single row (R53) and fits wherever its two letters
stand: one interneuron for every `th` in the stream (D11).

**No variable forms in this stream.** Two functions that differ at a node, `{h, t@−1}` and `{h, s@−1}`, are
generalized into `{h, C@−1}`, `C` a class whose members are `t` and `s`, and the general function is priced as a
replacement of the two (R52). The two cost 6 in the dictionary, three each, and 2 per occurrence. The class
costs 5, one relation, `h`, the class at its node and two members, and 3 per occurrence, since the member that
stood is written as a value (D13). Each specific function has at least two occurrences, so the class is never
cheaper, and it does not join. The dictionary holds one function per pair of letters that recurs, and nothing
general.

# 3. Words

A word is not one neuron. What the machine names is the stretch that recurs, a letter at a time.

Take `the`. `TH` fires at `h`'s frame `f`. At `f + 1`, `e` fires. Its neighborhood holds `TH` at `−1` as
itself, and `h` under it at `−1` (D5, D47); `t` stands at `−2`, beyond its reach. The call returns two
relations, `(e, TH, −1)` and `(e, h, −1)` (R21).

- `{e, h@−1}`, held in two rows, is a function. On the board it lands on one of the two activations `TH`
  covers, and is credited nothing for it (R23): it measures 2 for 2 and is not taken.
- `{e, TH@−1}`, once `TH` has stood before `e` in two rows, is a function. It lands on everything `TH`'s call
  covers, so it takes that call back (R23): A is `e`'s record and `TH`'s record, 4; B is 2. The election takes
  it. `TH`'s call leaves the file, `TH`'s activation leaves the apex, and the new interneuron, `THE`, fires at
  `e`'s frame, standing for `t h e` and expanding to it (R28). `h`, open at age 1, connects once more, to `THE`
  (D10); `t`'s window has closed (D9).

So each letter's function that pays names what stood on the apex a frame before: the bare letter, or the
interneuron that covered it, and through that interneuron the whole stretch it stands for. The stretch grows a
letter per frame for as long as each next letter has stood after that interneuron twice. Where it has not, the
relation is in one row, nothing is built on it (R21), the letter stands bare, and the chain starts again from the
next pair that recurs.

An interneuron sees `R` beyond what it stands for (D4), so it also sees the bare letter or the interneuron just
before its stretch. A function from its own history may name that, where it recurs twice, and join two stretches
into one.

**One word, several interneurons.** Which stretch an interneuron stands for is decided by what recurred twice
first. After a space, `{t, space@−1}` recurs early; its interneuron covers the space and `t`, `h`'s function
names that interneuron, `e`'s names the one over `h`, and the interneuron at `e` stands for ` the`. Where `n`
stands before the space and `{space, n@−1}` recurred first, the chain that reaches `e` stands for `n the`; after
`of`, for `of the`. These are different stretches. The index gives one interneuron to a function whose expansion
comes round to what an interneuron already stands for, over exactly that length and not from where the call
would stand (D36, R43), and none of these does. So `the` has one interneuron after a space, another in `n the`,
another in `of the`. The machine compresses the stretch that recurs, not the word.

A stretch of any length forms this way, and nothing has to be a power of two long: a bare letter is on the apex,
and whatever fires next within reach can name it. A long stretch does not see far for being long: an interneuron
standing for eight letters sees one beyond them, eight back from where it fires (D4).

The radius sets how much of this happens at the base. At `R = 4` a letter sees four letters back, each at its
own offset (D6), so a stretch of up to five letters is one function from its last letter's history, and the
chain above starts from there; [algorithm-xy.md](algorithm-xy.md) is a stream declared that way.

# 4. A doubled letter

`ll`, `oo` and `ss` are pairs like any other: the second `l` sees the first one back, a neuron being its own
neighbor at any offset but zero (D26). `{l, l@−1}` has two fixed nodes and one relation, so its entry is 3, it
saves 2 per row, and its margin is positive at two rows, as `{h, t@−1}` is. The same holds for `oo` from `o`'s
history and `ss` from `s`'s. A doubling that recurs is covered by its own interneuron. One too rare to recur
within `l`'s history stays as two letters on the apex, as any rare pair does.

# 5. What the apex holds after a sentence

Interneurons for the stretches that recur, each at the frame of its last letter; the letters nothing covered;
and the spaces, which are letters like any other until a function names them. The apex is what the next letter
sees (D5), and what connects to what follows (D25): every uncovered activation open at the next frame connects
to the letter that stood there (R31). It is also what votes (R27): each activation proposes at the frame ahead
the letters its connections name, and the letter the most voters place is output, weakly, for the report to
replace (R47, R40).
