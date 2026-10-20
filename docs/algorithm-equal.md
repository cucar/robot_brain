# If equal, the third, else the fourth, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that is shown four digits `a, b, c, d` in one frame, is
asked, and answers `c` where `a` and `b` are the same digit and `d` where they are not —
`f(a, b, c, d) = a == b ? c : d` — written out neuron by neuron. Nothing here is normative, and the machine is
shown running: its tables and connections are as listed, and its histories already hold requests like this
one. It shows the one equality test the design has, a parameter over two positions (D44); an answer that is a
copy of an argument, a parameter again, bound from what was shown and placed at what is said (D44, R28); and a
vote in which the one voter that knows the situation outweighs the digits, which know only their own value.

---

# 1. The environment

**In.** One channel, laid out over time and one spatial dimension, `slot`, with five positions, `0` to `4`.
The radius is `4` in `slot`, so every position is told from every other exactly, and `1` in time, so a base
neuron sees one frame back (D1, D6). It declares two event dimensions: `digit`, with ten buckets, `0` to `9`,
and `run`, with one bucket, `run`. A request is one frame: `a`, `b`, `c` and `d` in `digit` at slots `1` to
`4`, and `run` at slot `0`.

**Out.** One action dimension in the same channel, `say`, with one base action, `say`. The answer is `say` at
slot `0` of the frame after the request, with a digit beside it in `digit` at the same slot: the digit is an
event the machine expects there, and both are its output (§3.5, R40). A frame of silence follows the answer,
so nothing of one request is within reach of the next.

**Policy.** Every entry of both tables of the neighborhood policy is on (D50): anything may be a neighbor of
anything, and anything may connect to anything. The action policy is nothing (D51): where every candidate in
`say` is negative, the machine says nothing.

**Teaching.** While the machine is being taught, the world runs `say` and shows the right digit beside it in
the frame after each request, and rewards it, every lesson the same (§3.5). It mixes equal and unequal requests
and varies every digit. Afterwards it runs nothing, executes what the machine outputs, and pays what was right
and what was wrong.

**Why `a` and `b` are two positions of one dimension.** The only thing in the design that tests two places for
the same content is a parameter, and what it tests is that one neuron stands at both (D44). A neuron is a
dimension–bucket pair (D2), so `3` in a dimension `a` and `3` in a dimension `b` are two neurons that have
nothing in common, and no pattern could say they are equal short of a function per digit naming both outright,
which is the diagonal memorized and not equality learned. Two positions of one dimension, in space as here or
in time, is where a parameter holds.

**Why the digits stand bare.** Each of the four slots holds a different digit from one request to the next, so
each is a class candidate in `run`'s table (D41). With twelve base neurons a class of the ten digits saves
`log₂ 12 − log₂ 10`, about a quarter of a bit, each time it covers one, against a line that names ten neurons;
the history is kept short of the length at which that pays (D30, R4), and no class forms. The digits are
therefore neighbors in their own right wherever they are within reach, and they vote. Were the classes formed,
the digits would be covered and silent, and `say` would read them through the classes' value neurons instead
(D44, D45); the case shows the bare regime because it is the one a short history gives.

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `run`, and the ten digits |
| base action | `say` |
| the test | `P`, a parameter in `run`'s table, one offset of a slot: one and the same digit at both ends (D44); value neurons `P:0` to `P:9`, "the pair was k"; and `equal`, a function naming `run` and `P` a slot beside it, with its child `EQUAL`, which fires at slot `1`, the center of the three slots it covers (D48, D53) |
| the situation, seen from the answer | nothing of its own: `EQUAL` at slot `1` a frame back after an equal request, `run` at slot `0` after an unequal one, each a node of its own function below |
| the copies | `Pc`, a parameter in `say`'s table, one offset between slot `0` now and slot `3` a frame back: the answer is `c`; `Pd`, between slot `0` now and slot `4` a frame back: the answer is `d` (D44); value neurons `Pc:0` to `Pc:9` and `Pd:0` to `Pd:9` |
| the functions | two, in `say`'s table: `from-c`, naming `say`, `EQUAL` at slot `1` a frame back and `Pc`; `from-d`, naming `say`, `run` at slot `0` a frame back and `Pd`; their children `ANSWER-C` and `ANSWER-D` (D48) |

# 3. How it was learned

1. **The test is a parameter.** `run` fires at slot `0` and sees the four digits beside it at slots `1` to
   `4`, each at its own exact offset (D6). In every request where `a == b`, one and the same neuron stands at
   slots `1` and `2`: two offsets that hold one neuron is the relation a parameter is seeded on (D47), and its
   collapse makes it `P`, one offset of a slot (D27). Agreement is its whole test, so a digit never seen equal
   before is `P`'s value the first time it is, and a member from then on. `P` alone covers `a` and `b` and not
   `run`; `run` with `P` holding a slot beside it recurs in every equal request, so the collapse grows a
   function over the two, `equal` (D27, D38).
2. **The branch is whether `P` holds.** In an equal request `equal`'s bid covers `run`, `a` and `b` for the
   price of one member, and is taken: its child `EQUAL` fires at slot `1`, the center of the three slots it
   covers (D53), `P`'s value neuron `P:k` fires over `P`'s two positions, at slot `1` on the tie (D45), and `c`
   and `d` stand bare beside them. In an unequal request `P`'s two positions disagree, its value is the digit at
   the later position and the other is a failed neighbor (D44), the bid no longer covers more than it costs, and
   nothing is taken: `run` and all four digits stand bare. Two situations, and no neuron ever says which digits
   they held beyond `P:k`'s own.
3. **The answer sees the situation by what stands.** `say` fires at slot `0` of the next frame and sees the
   request's frame. After an equal request `EQUAL` stands at slot `1` a frame back; after an unequal one `run`
   stands at slot `0`. They stand at different slots, so no class joins them, and each is a node of a function
   of its own below.
4. **The copies are parameters.** In every equal request the digit at slot `3` a frame back is the digit
   beside `say` now, and in every unequal one the digit at slot `4` is: two pairs of offsets that hold one
   neuron, and two parameters, `Pc` and `Pd`, each over the slot beside `say` now and a slot of the frame
   before (D47, D27). Each writes which member once and covers the answer in full. A parameter is open on its
   members, so a digit never answered before is copied the first time it is shown.
5. **The functions are grown over them.** `say`, `EQUAL` a frame before it at slot `1` and `Pc` hold together
   in every equal row, `say`, `run` at slot `0` and `Pd` in every unequal one, and the collapse grows a
   function over each three (D33, D27): `from-c` and `from-d`, with children `ANSWER-C` and `ANSWER-D`. A
   `from-c` bid covers `say`, `EQUAL`, `c` and the answer, slots `0` to `3` across two frames, so `ANSWER-C`
   fires at slot `1` of the answer's frame, the center of the box (D53); `from-d` covers `run` and `d` where
   `from-c` covered `EQUAL` and `c`, slots `0` to `4`, so `ANSWER-D` fires at slot `2`. A copy's value neuron
   fires over its two positions, at slot `1` and slot `2` by the same rule (D45). Which is taken is
   which parameter holds: the other's two positions disagree, so it takes the digit at its first position as its
   value and is charged a failed neighbor at the other (D44), and the bid that holds covers more for less and is
   taken first (D22, D28).
6. **The situation learns its answer.** In the taught frames `ANSWER-C` fired over `EQUAL` and `ANSWER-D` over
   `run`. Each is covered in the frame the answer runs, and connects once to the neuron that covered it, with
   that frame's reward (D10): `EQUAL` holds `ANSWER-C` at one frame ahead, `run` holds `ANSWER-D`. `P:k`, bare
   at slot `1`, connects to `ANSWER-C` at its own offset like any uncovered activation (R31). Before the
   functions existed, `say` and the digit stood bare at the answer frame and both connected to them directly; those
   connections stay (R31) and place the same things.
7. **The digits learn both.** A digit at slot `3` is covered by `ANSWER-C` after an equal request and stands
   bare beside `ANSWER-D` after an unequal one, and connects to each at the offset of its own slot (D25,
   R31). A digit's identity is its type alone (D11), so the neuron `8` holds `ANSWER-C` and `ANSWER-D` at the
   offset from slot `3`, both at the offset from slot `4`, and `ANSWER-D` at the offsets from slots `1` and
   `2`, where `a` and `b` stand bare only in unequal requests. After an equal request `a` and `b` are covered at
   age 0 and write nothing (D10).

# 4. One request, frame by frame, after teaching

`3, 3, 8, 5`. The answer is `8`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `run`, `3`, `3`, `8`, `5` | `EQUAL` at slot `0`, `P:3` at slot `1`, `8`, `5` | `equal` holds and covers `run`, `a` and `b`. Each apex activation infers at its age 0 (R36). The vote on frame 2 is below. |
| 2 | `say` reported as run; nothing reported in `digit` | `ANSWER-C` at slot `1`, over `say`, `EQUAL`, `8` and the expected `8` | The expected `8` stands, since nothing replaced it (D40). Everything open and uncovered connects to `ANSWER-C` with the reward, `P:3` among them; `EQUAL` and the `8` at slot `3`, covered, connect once to it (D10). |
| 3 | silence | nothing within reach | |

**The vote at frame 1**, voter by voter. Each reads its connections and the machine expands those that land
at frame 2, binding each parameter from what stands at the request (R28, R45). The match (R49) reads each
inference's line against frame 1: `EQUAL` stands at slot `1` where `from-c` names it, and each copy
parameter's earlier position holds a digit, so every inference below is proposed.

| Voter | its connections landing at slot `0` of frame 2 | what each places there |
|---|---|---|
| `EQUAL` | `ANSWER-C` | `say`, and `Pc` bound from slot `3`: an `8` |
| `P:3` at slot `1` | `ANSWER-C`, at the offset from slot `1` | `say`, and an `8` |
| the `8` at slot `3` | `ANSWER-C` and `ANSWER-D`, both at the offset from slot `3` | `say` and an `8`; `say` and `Pd` bound from slot `4`: a `5` |
| the `5` at slot `4` | `ANSWER-C` and `ANSWER-D`, both at the offset from slot `4` | `say` and an `8`; `say` and a `5` |

A digit's connections from the other slots land at other positions, where nothing is resolved, and propose
nothing (R46). The two `3`s are covered and are not heard (D10).

Every estimate is the teacher's reward, so the candidates tie on estimate and the larger share of voters wins
(R47). `say` is placed by every voter and is output. In `digit` at slot `0`, `8` takes `EQUAL` and `P:3` whole
and half of each digit, three shares; `5` takes half of each digit, one share. `8` is expected, and output. The
digits are not silenced; they hedge, and the voters that know the situation decide.

`3, 4, 8, 5` is the mirror. Nothing covers anything at frame 1, so the voters are `run`, `3`, `4`, `8` and
`5`. `run` holds `ANSWER-D` alone and places a `5`; the `3` and the `4`, at slots `1` and `2`, hold `ANSWER-D`
alone at those offsets and place a `5`; the `8` and the `5` split. `5` wins by four shares to one.

**With the classes formed.** A wider alphabet, or a longer history, makes a digit class pay (D30). `run`'s
table then holds `P` and four classes, `A`, `B`, `C` and `D`, one per slot (D41), and two functions over them:
`same`, naming `run`, `P`, `C` and `D`, and `differ`, naming `run`, `A`, `B`, `C` and `D` (D38); `equal` has
re-centered into `same` (D29). A class stands only where a function names it (D41), so the classes reach the
apex only inside those functions. Both children cover slots `0` to `4` and fire at slot `2`, the center (D53),
so in `say`'s table a class `Q` of `same` and `differ` now pays, named at slot `2` a frame back by `from-c` and
`from-d` in place of `EQUAL` and `run`; `Pc` reads `c` through `C`'s value neuron as `C`'s value at slot `3`
(D44, D45), and a `from-c` bid covers that value neuron itself.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `run`, `3`, `3`, `8`, `5` | `same` at slot `2`, `P:3` at slot `1`, `C:8` at slot `3`, `D:5` at slot `4` | `same` covers `run` and the four digits, and its value neurons stand where their members stood. Each infers at its age 0 (R36). |
| 2 | `say` reported as run; nothing reported in `digit` | `ANSWER-C`, over `say`, `same`, `C:8` and the expected `8` | `same`, covered through `Q`, and `C:8`, covered through `Pc`, connect once to `ANSWER-C` (D10); `P:3` and `D:5`, bare, connect to it too (R31). |

The vote at frame 1, voter by voter: `same` and `P:3` each hold `ANSWER-C` alone, since both stand only in
equal requests. `C:8` holds `ANSWER-C`, which covered it after every equal request, and `ANSWER-D`, beside
which it stood bare after every unequal one; `D:5` holds the mirror. So `8` takes two voters whole and half of
two more, three shares, and `5` one. The digits say nothing; their value neurons hedge in their place, and the
branch voters win, as in the bare regime.

# 5. The price

The alphabet is the twelve base neurons here, so a base symbol costs `log₂ 12 ≈ 3.6` bits. `run`'s table holds
one pattern, so which of them costs nothing, and `P` chooses a member at `log₂ 10 ≈ 3.3` (D13).

| the request, written as | bits |
|---|---|
| flat: five base symbols | 17.9 |
| `equal`, equal request: `run`, one member; `c` and `d` as themselves | 14.1 |
| `equal`, unequal request: `run`, one member, one position failed; `a` not covered | covers 7.2 for 7.9 — not taken |

`say`'s table holds four patterns, so which of them costs `2`, and a copy parameter chooses among ten at `3.3`.
A `from-c` call covers `say`, `EQUAL` at what its own call cost, `6.9`, the digit `c` and the answer, `17.7`
bits in all, for `say`, which pattern and one choice, `8.9`: it saves about `9` bits on an equal request.
`from-d` on an unequal request covers `run` where `from-c` covered `EQUAL`, and saves about `5.5`. Either
saving is positive, which is all the election asks (R22, R24).

# 6. What this case leans on

- **The digits stand bare.** No class in `run`'s table pays within the history, so the digits are on the apex
  when `say` fires, the copy parameters are seeded on them directly, and they vote. Were the alphabet wide
  enough for a digit class to pay, `P`'s bid would cover `c` and `d` through it, what stood at their slots would
  be the classes' value neurons, and the copy parameters would read the digits through them (D44): the copy is
  the same, the digits are silent, and the vote at the request is `EQUAL` or `run`, with `P:k` beside the
  first. The case shows the bare regime; the remarks say what the other looks like.
- **One frame, so one offset.** Laid out across frames, a digit shown third would learn the answer three frames
  after it and shown fourth two frames after, and would place the answer at both distances whatever it was
  shown as now. In one frame every digit sees the answer one frame after itself, and the slot is in the
  connection's offset, so what a digit learned at one slot lands where that slot's answer goes and nowhere
  else (D25, R28).
- **A majority with tied estimates.** The branch is read at the request by nothing but the vote, and the vote
  is right because `EQUAL` or `run` names one answer while each digit names both. Estimates do not separate them
  while the teacher pays every lesson the same (R47). What pulls the wrong function's child down afterwards is the
  machine's own errors: a `5` said in an equal request is paid negatively, the connections that placed it take
  the share, and the mean falls (R31, R32).
- **Teaching is mixed.** Taught equal requests first, `run` would hold functions naming `a` and `b` outright
  before `P` had two rows to be seeded on; `P` forms from the rows the functions fail, as the general form does
  in [algorithm-xy.md](algorithm-xy.md), and nothing is lost but time.

# 7. The same thing as code

| in the code | in the machine |
|---|---|
| `a == b` | `P` holding at slots `1` and `2`: one neuron at both |
| the two arms of the conditional | `from-c` and `from-d`, two functions in `say`'s table, taken by which copy holds |
| `return c` | `Pc`, bound from slot `3` and placed beside `say` |
| choosing the arm | no neuron: `EQUAL` or `run` names the child it was followed by, the digits name both, and the share decides (R47) |
| the call site that knows the whole | `ANSWER-C` and `ANSWER-D`, one frame after the request, each with the value neurons of `Q` and its copy beside it |
