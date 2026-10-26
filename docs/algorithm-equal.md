# If equal, the third, else the fourth, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that is shown four digits `a, b, c, d` in one frame, is
asked, and answers `c` where `a` and `b` are the same digit and `d` where they are not,
`f(a, b, c, d) = a == b ? c : d`, written out neuron by neuron. Nothing here is normative, and the machine is
shown running: its dictionary and its connections are as listed, and its histories already hold requests like
this one. It shows the one equality test the design has, a parameter over two nodes (D44); an answer that is a
copy of an argument, a parameter again, bound from what was shown and placed at what is said (D44, R28); and
a vote in which the one voter that knows the situation outweighs the digits, which know only their own value.

---

# 1. The environment

**In.** One channel, laid out over time and one spatial dimension, `slot`, with five positions, `0` to `4`.
The radius is `4` in `slot`, so every position is told from every other exactly, and `1` in time, so a base
neuron sees one frame back (D1, D6). It declares two event dimensions: `digit`, with ten buckets, `0` to `9`,
and `run`, with one bucket, `run`. A request is one frame: `a`, `b`, `c` and `d` in `digit` at slots `1` to
`4`, and `run` at slot `0`.

**Out.** One action dimension in the same channel, `say`, with one base action, `say`. The answer is `say` at
slot `0` of the frame after the request, with a digit beside it in `digit` at the same slot: the digit is an
event the machine infers there, and both are its output (§3.3, R40). A frame of silence follows the answer,
so nothing of one request is within reach of the next.

**Policy.** Every entry of both tables of the neighborhood policy is on (D50): anything may be a neighbor of
anything, and anything may connect to anything. The action policy is nothing (D51): where every candidate in
`say` is negative, the machine says nothing.

**Teaching.** While the machine is being taught, the world runs `say` and shows the right digit beside it in
the frame after each request, and rewards it, every lesson the same. It mixes equal and unequal requests and
varies every digit. Afterwards it runs nothing, executes what the machine outputs, and pays what was right and
what was wrong.

**Prices.** The layout has one spatial dimension, so a bare record is three numbers: the neuron, the frame and
the slot (D13). A call's record is three numbers, then one per relation that did not come true, one per value,
and one per position of an activation it lands on that no relation fixes (D12, D22).

**Why `a` and `b` are two positions of one dimension.** The only thing in the design that tests two places for
the same content is a parameter, and what it tests is that one neuron stands at both (D44). A neuron is a
dimension–bucket pair (D52), so `3` in a dimension `a` and `3` in a dimension `b` are two neurons that have
nothing in common, and no function could say they are equal short of a function per digit naming both outright.
That is the diagonal memorized, not equality learned. Two positions of one dimension, in space as here or in
time, is where a parameter holds.

**Why `c` and `d` stand bare.** A function that names the equal pair and the digit at slot `3` is grouped only
where one such request recurs twice within `run`'s history (R50). With ten digits that is rare, and a class
at slot `3` is made only from two such functions (R52). So after a request the digits at slots `3` and `4`
stand on the apex bare, at three numbers each. Bare or bound, a digit stays on the apex and votes (D10).

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `run`, and the ten digits |
| base action | `say` |
| the test | `equal`, a function naming `run` and a parameter `P` at slot `+1`, at slot `+2` and at time `0` from it: one and the same digit at both slots (D44); its interneuron `EQUAL` (D53) |
| the situation, seen from the answer | nothing of its own: `EQUAL` stands a frame back at slot `1` after an equal request and at slot `0` after an unequal one, and each answer function names it where it stands |
| the answer | two functions: `from-c`, naming `say`, `EQUAL` at `(−1, +1)`, and a parameter `Pc` at `(−1, +3)` and at `(0, 0)` from `say`, the answer is `c`; `from-d`, naming `say`, `EQUAL` at `(−1, 0)`, and `Pd` at `(−1, +4)` and at `(0, 0)`, the answer is `d` (D44); their interneurons `ANSWER-C` and `ANSWER-D` (D48) |

Offsets are written `(time, slot)`.

# 3. How it was learned

1. **A digit's own equality is grouped first.** `run` fires at slot `0` and sees the four digits beside it at
   slots `1` to `4`, each at its own exact offset (D6). Its activate call returns one relation per neighbor per
   dimension (R21): `3` at slot `+1`, `3` at slot `+2`, `3` at time `0`, and so on for the other digits. In the
   rows where `a` and `b` are both `3`, those three relations are held together, and grouping merges them
   (R50): `{run, 3@s+1, 3@s+2, 3@t0}`. A relation lands on one activation (D15). The relation at time `0`
   lands on the `3` at slot `1`, the smaller coordinate (D31). The other `3`'s frame is fixed by no relation,
   so the call writes it. The function writes two records of `3` and `run`'s, nine numbers, as one call of four:
   it saves five per row. Its entry is three relations and two neurons, five, so its margin is positive from
   its second row (D30).
2. **Two digits' equality is one parameter.** When `{run, 4@s+1, 4@s+2, 4@t0}` is grouped, find reads each of
   its relations with the digit end open (R53) and reaches the function for `3`. The two differ only at nodes
   that hold one neuron within each function, so those nodes are one parameter (R52): `equal`, which is `{run,
   P@s+1, P@s+2, P@t0}`. Its entry is three relations and `run`, four. Its call is three numbers, one value,
   and the frame of the digit at slot `2`, five. It is priced as a replacement of the two specific functions:
   two entries of five and four per row, against one entry of four and five per row. With two rows each it
   writes `run`'s history for twenty-four where the specifics write it for twenty-six. It joins, and the two
   leave. From then on a digit's equality
   is read through `equal`'s open ends, and no specific function is grouped again (R53).
3. **Agreement is the whole test.** A parameter is open on its members (D44): a digit never seen equal before
   is `P`'s value the first time it stands at both slots, and a member from then on.
4. **`equal` fits an unequal request too, with one relation that did not come true.** `P`'s value is the digit
   at the earliest of its nodes, slot `1` (D44). The relation at slot `+2` does not come true, and the relations
   at slot `+1` and time `0` land on the digit at slot `1`. The call covers `run` and that digit, six numbers,
   for a record of five: three, one value, one relation. It is taken (R24). So `run` is never bare after a
   request: `EQUAL` stands over it either way, at slot `1` after an equal request, the centroid of slots `0`,
   `1` and `2`, and at slot `0` after an unequal one, the centroid of `0` and `1` rounded half toward the
   smaller (D53). Two situations, told apart by where `EQUAL` stands.
5. **The answer sees the situation.** `say` fires at slot `0` of the next frame and sees the request's frame.
   After an equal request it sees `EQUAL` at `(−1, +1)`, `c` at `(−1, +3)`, `d` at `(−1, +4)`, `a` and `b`
   bound under `EQUAL`, and the digit the world shows beside it at `(0, 0)`. In the rows where `c` is `8`, the
   relations to `EQUAL`, to the `8` at slot `3` and to the `8` beside `say` are held together, and grouping
   gives `{say, EQUAL@(−1, +1), 8@(−1, +3), 8@(0, 0)}`: six relations, two per node. It covers `say`, `EQUAL`'s
   record of five and two records of `8`, fourteen numbers, for three. The function for `5` reaches it through
   the open ends, the differing nodes hold one neuron within each, and R52 makes `from-c`, with `Pc` tying
   slot `3` of the request to slot `0` of the answer. The same over unequal requests makes `from-d`, naming
   `EQUAL` at `(−1, 0)` and `Pd` from slot `4`. The two differ in an offset, not at a node, so nothing
   generalizes them further.
6. **A copy is a parameter.** `Pc` writes its digit once, as `from-c`'s one value, and both of its nodes are
   covered by that (D13). A parameter is open on its members, so a digit never answered before is copied the
   first time it is shown.
7. **The situation learns its answer.** In the frame `say` runs, `ANSWER-C` or `ANSWER-D` covers `EQUAL`, and
   `EQUAL` connects once to the neuron that covered it, with that frame's reward (D10): `ANSWER-C` at
   `(+1, 0)`, from slot `1` to slot `1`, and `ANSWER-D` at `(+1, +1)`, from slot `0` to slot `1`. Before the
   answer functions existed, `say` and the digit stood bare at the answer frame, and `EQUAL` connected to
   them directly; those connections stay (R31) and place the same things.
8. **The digits learn both.** Bound or bare, a digit stays on the apex and connects to what stands on the
   next frame's apex (D10, R31): `ANSWER-C` after an equal request and `ANSWER-D` after an unequal one, each at
   the offset from the digit's slot to slot `1`. A digit's identity is its type alone (D11), so the neuron `8`
   holds both at `(+1, −2)` from slot `3`, both at `(+1, −3)` from slot `4`, and both at `(+1, 0)` and
   `(+1, −1)` from slots `1` and `2`.

# 4. One request, frame by frame, after teaching

`3, 3, 8, 5`. The answer is `8`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `run`, `3`, `3`, `8`, `5` | `EQUAL` at slot `1`; `3`, `3` bound at slots `1` and `2`; `8`, `5` bare | `equal` covers `run` and binds `a` and `b`, nine numbers for five. Each apex activation infers at its age 0 (R36). The vote on frame 2 is below. |
| 2 | `say` reported as run; nothing reported in `digit` | `ANSWER-C` at slot `1`, over `say` and `EQUAL`; the `8` at slot `3` and the inferred `8` at slot `0` bound beside it | The inferred `8` stands, since nothing replaced it (D40). Everything open on the apex connects to `ANSWER-C` with the reward, the four digits among them; `EQUAL`, covered, connects once to it (D10). |
| 3 | silence | nothing within reach | |

**The vote at frame 1**, voter by voter. Each reads its connections at age 0, and the machine expands what
lands at frame 2, binding each parameter from what stands at the request (R28, R45). The match (R49) reads
each inference's function against frame 1. `run` is covered and is not heard (D10).

| Voter | its connections placing a call at slot `1` of frame 2 | what the call places at slot `0` |
|---|---|---|
| `EQUAL` at slot `1` | `ANSWER-C` at `(+1, 0)` | `say`, and `Pc` bound from slot `3`: an `8` |
| the `3` at slot `1`, bound | `ANSWER-C` and `ANSWER-D`, both at `(+1, 0)` | `say` and an `8`; `say` and `Pd` bound from slot `4`: a `5` |
| the `3` at slot `2`, bound | the same, at `(+1, −1)` | the same |
| the `8` at slot `3` | the same, at `(+1, −2)` | the same |
| the `5` at slot `4` | the same, at `(+1, −3)` | the same |

For each of these the match credits `EQUAL` where the function names it, five numbers, and the digit the
parameter lands on, three, and charges nothing, so every one is proposed.

Every estimate is the teacher's reward, so the candidates tie on estimate and the larger share of voters wins
(R47). `say` is placed by every voter and is output. In `digit` at slot `0`, `8` takes `EQUAL` whole and half
of each of the four digits, three shares; `5` takes half of each digit, two shares. `8` is inferred, and output.
The digits are not silenced, bound or bare; they hedge, and the one voter that knows the situation decides.

`3, 4, 8, 5` is the mirror. `EQUAL` stands at slot `0`, over `run` and the `3`. Its connection at `(+1, +1)`
places `ANSWER-D` at slot `1`: `say` at slot `0`, and `Pd` bound from slot `4`, a `5`. Each digit holds both
and splits. `5` wins by three shares to two.

**The other slots.** A digit's connections from the other slots it has stood at place `ANSWER-C` and
`ANSWER-D` at slots `2`, `3` and `4` as well, and `EQUAL` at slot `1` places `ANSWER-D` at slot `2` through its
connection at `(+1, +1)`. Read back from there, each such call names `EQUAL` at a slot where it does not stand:
`EQUAL` stands within reach at another offset, so the match credits its record and charges one offset, and a
parameter node that lands outside the layout charges its relation (R49). The credit exceeds the charge, so each
is proposed, and `say` with a digit is placed at slots `1` to `3` of frame 2 too, each resolved at its own
position (R47). The world's reward on those slots decides whether those connections keep placing them.

# 5. The price

| the request, priced as | numbers |
|---|---|
| flat: five bare records | 15 |
| through `equal`, equal request: `EQUAL`'s record, three, `P`'s value, one, and the frame of `b`, one; `c` and `d` bare, six | 11 |
| through `equal`, unequal request: `EQUAL`'s record, its value, and one relation that did not come true, five; `b`, `c` and `d` bare, nine | 14 |

On the answer frame after an equal request, `from-c` covers `say`, `EQUAL`'s record of five, `c` and the
digit beside `say`, fourteen numbers, for four: `ANSWER-C`'s record and `Pc`'s value. `from-d` on the same
frame names `EQUAL` at slot `0`, where it does not stand, and `Pd` at slot `4`, where a `5` stands against the
`8` beside `say`. Three relations do not come true, and `EQUAL`'s slot is written in the call. It covers eleven
for eight. `from-c` is taken first, and `from-d` then covers nothing that is left (R24). After an unequal
request `EQUAL`'s record is five as well, and `from-d` covers fourteen for four.

# 6. What this case leans on

- **The digits stand on the apex, bare or bound.** The copy parameters are grouped on them directly, and they
  vote. A bound digit is the same activation at the same place (D10), so nothing about the digits changes for
  `say` whether `EQUAL` binds them or not.
- **One frame, so one offset.** Laid out across frames, a digit shown third would learn the answer three frames
  after it and shown fourth two frames after, and would place the answer at both distances whatever it was
  shown as now. In one frame every digit sees the answer one frame after itself, and the slot is in the
  connection's offset, so what a digit learned at one slot lands where that slot's answer goes (D25, R28).
- **A majority with tied estimates.** The branch is read at the request by nothing but the vote, and the vote
  is right because `EQUAL` names one answer at the offset it stands at while each digit names both. Estimates
  do not separate them while the teacher pays every lesson the same (R47). What pulls the wrong answer down
  afterwards is the machine's own errors: a `5` said in an equal request is paid negatively, the connections
  that placed it take the share, and their estimates fall (R31, R32).
- **Teaching is mixed.** Taught equal requests first, `run` would group the same per-digit functions, and
  `equal` would replace them the same way; what changes is only when `EQUAL` first stands at slot `0`.

# 7. The same thing as code

| in the code | in the machine |
|---|---|
| `a == b` | `P` holding at slots `1` and `2`: one neuron at both |
| the two arms of the conditional | `from-c` and `from-d`, two functions, taken by where `EQUAL` stands and which copy parameter holds |
| `return c` | `Pc`, bound from slot `3` and placed beside `say` |
| choosing the arm | no neuron: `EQUAL` names the answer it was followed by at the offset it stands at, the digits name both, and the share decides (R47) |
| the call site that knows the whole | `ANSWER-C` and `ANSWER-D`, one frame after the request, each with the digits it bound beside it |
