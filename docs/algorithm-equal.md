# If equal, the third, else the fourth, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that is shown four digits `a, b, c, d` in one frame, is
asked, and answers `c` where `a` and `b` are the same digit and `d` where they are not —
`f(a, b, c, d) = a == b ? c : d` — written out neuron by neuron. Nothing here is normative, and the machine is
shown running: its tables and connections are as listed, and its histories already hold requests like this
one. It shows the one equality test the design has, a parameter over two places (D44); an answer that is
a copy of an argument, a parameter again, bound from what was shown and placed at what is said (D44,
R28); and a vote in which the one voter that knows the situation outweighs the digits, which know only their
own value.

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
each is a class candidate in `run`'s table (D41). With twelve base neurons a class over the ten
digits saves `log₂ 12 − log₂ 10`, about a quarter of a bit, each time it binds one, against a class that names
ten neurons; the history is kept short of the length at which that pays (D30, R4), and no class forms. Bound
or bare, a digit stays on the apex and votes (D10); what the classes change is which functions stand over
the request. The case shows the bare regime because it is the one a short history gives.

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `run`, and the ten digits |
| base action | `say` |
| the test | `equal`, a function in `run`'s table naming `run` and a parameter `P` tying slots `1` and `2`: one and the same digit at both (D44); its child `EQUAL`, which fires at slot `0`, `run`'s coordinate (D48, D53) |
| the situation, seen from the answer | nothing of its own: `EQUAL` at slot `0` a frame back after an equal request, `run` at slot `0` after an unequal one, each a node of its own function below until a class over the two pays |
| the functions | two, in `say`'s table: `from-c`, naming `say`, `EQUAL` at slot `0` a frame back, and a parameter `Pc` tying slot `0` now to slot `3` a frame back, the answer is `c`; `from-d`, naming `say`, `run` at slot `0` a frame back, and `Pd` tying slot `0` now to slot `4` a frame back, the answer is `d` (D44); their children `ANSWER-C` and `ANSWER-D` (D48) |

# 3. How it was learned

1. **The test is a parameter.** `run` fires at slot `0` and sees the four digits beside it at slots `1` to
   `4`, each at its own exact offset (D6). In every request where `a == b`, one and the same neuron stands at
   slots `1` and `2`: a same-to-same relation (D34, D47). The pick grows a function from it over `run`,
   `equal`, whose collapse ties the two slots in a parameter `P` (D33, D27). Agreement is its whole test,
   so a digit never seen equal before is `P`'s value the first time it is, and a member from then on.
2. **The branch is whether `P` holds.** In an equal request `equal`'s bid covers `run` and binds `a` and `b` for
   the price of one member, and is taken: its child `EQUAL` fires at slot `0`, `run`'s coordinate (D53), and
   `a`, `b`, `c` and `d` all stand on the apex, the first two bound (D10). In an unequal request `P`'s two
   places disagree, its value is the digit at the later place and the other is a failed node (D44), the bid no
   longer covers more than it costs, and nothing is taken: `run` and all four digits stand bare. Two
   situations, told apart by what stands at slot `0`.
3. **The answer sees the situation by what stands.** `say` fires at slot `0` of the next frame and sees the
   request's frame. After an equal request `EQUAL` stands at slot `0` a frame back; after an unequal one `run`
   stands there. Two neurons taking turns at one offset from `say` is a node-to-open relation (D34, D47), so a
   class of the two is a candidate, and it pays once the history is long, as the digit classes do (D30). In
   the bare regime it does not, and each is a node of a function of its own below.
4. **The copies are parameters.** In every equal request the digit at slot `3` a frame back is the digit
   beside `say` now, and in every unequal one the digit at slot `4` is: two same-to-same relations in `say`'s
   table (D47). The pick grows a function from each over `say`, and each collapse ties the two places in a
   parameter, `Pc` and `Pd` (D33, D27), costs which member once, and covers the answer in full. A
   parameter is open on its members, so a digit never answered before is copied the first time it is
   shown.
5. **The functions find the situation.** The collapse of the function with `Pc` finds `EQUAL` a frame before
   `say` at slot `0` in every use, and adds it as a fixed node; the one with `Pd` finds `run` there. So `say`'s
   table holds `from-c` and `from-d`, with children `ANSWER-C` and `ANSWER-D` (D27). A `from-c` bid covers
   `say` and `EQUAL` and binds `c` and the answer, and `ANSWER-C` fires at slot `0` of the answer's frame,
   `say`'s coordinate (D53); `from-d` covers `run` where `from-c` covered `EQUAL` and binds `d`, and `ANSWER-D`
   fires at the same place. Which is taken is which parameter holds: the other's two places disagree, so it takes
   the digit at its first place as its value and is charged a failed node at the other (D44), and the bid that
   holds covers more for less and is taken first (D22, D28).
6. **The situation learns its answer.** In the taught frames `ANSWER-C` fired over `EQUAL` and `ANSWER-D` over
   `run`. Each is covered in the frame the answer runs, and connects once to the neuron that covered it, with
   that frame's reward (D10): `EQUAL` holds `ANSWER-C` at one frame ahead, `run` holds `ANSWER-D`. Before the
   functions existed, `say` and the digit stood bare at the answer frame and both connected to them directly;
   those connections stay (R31) and place the same things.
7. **The digits learn both.** Bound or bare, a digit stays on the apex and connects (D10): a digit at slot `3`
   connects to `ANSWER-C` after an equal request and to `ANSWER-D` after an unequal one, each at the offset of
   its own slot (D25, R31). A digit's identity is its type alone (D11), so the neuron `8` holds `ANSWER-C` and
   `ANSWER-D` at the offset from slot `3`, both at the offset from slot `4`, and both at the offsets from slots
   `1` and `2` as well, where `a` and `b` stand bound after an equal request and bare after an unequal one.

# 4. One request, frame by frame, after teaching

`3, 3, 8, 5`. The answer is `8`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `run`, `3`, `3`, `8`, `5` | `EQUAL` at slot `0`; `3`, `3` bound at slots `1` and `2`; `8`, `5` bare | `equal` holds, covers `run` and binds `a` and `b`. Each apex activation infers at its age 0 (R36). The vote on frame 2 is below. |
| 2 | `say` reported as run; nothing reported in `digit` | `ANSWER-C` at slot `0`, over `say` and `EQUAL`; the `8` and the expected `8` bound beside it | The expected `8` stands, since nothing replaced it (D40). Everything open on the apex connects to `ANSWER-C` with the reward, the four digits among them; `EQUAL`, covered, connects once to it (D10). |
| 3 | silence | nothing within reach | |

**The vote at frame 1**, voter by voter. Each reads its connections and the machine expands those that land
at frame 2, binding each parameter from what stands at the request (R28, R45). The match (R49) reads each
inference's function against frame 1: `EQUAL` stands at slot `0` where `from-c` names it, and each copy parameter's
earlier place holds a digit, so every inference below is proposed.

| Voter | its connections landing at slot `0` of frame 2 | what each places there |
|---|---|---|
| `EQUAL` | `ANSWER-C` | `say`, and `Pc` bound from slot `3`: an `8` |
| the `3` at slot `1`, bound | `ANSWER-C` and `ANSWER-D`, both at the offset from slot `1` | `say` and an `8`; `say` and `Pd` bound from slot `4`: a `5` |
| the `3` at slot `2`, bound | the same, at the offset from slot `2` | the same |
| the `8` at slot `3` | `ANSWER-C` and `ANSWER-D`, both at the offset from slot `3` | `say` and an `8`; `say` and a `5` |
| the `5` at slot `4` | `ANSWER-C` and `ANSWER-D`, both at the offset from slot `4` | `say` and an `8`; `say` and a `5` |

A digit's connections from the other slots land at other positions, where nothing is resolved, and propose
nothing (R46). `run` is covered and is not heard (D10).

Every estimate is the teacher's reward, so the candidates tie on estimate and the larger share of voters wins
(R47). `say` is placed by every voter and is output. In `digit` at slot `0`, `8` takes `EQUAL` whole and half
of each of the four digits, three shares; `5` takes half of each digit, two shares. `8` is expected, and
output. The digits are not silenced, bound or bare; they hedge, and the one voter that knows the situation
decides.

`3, 4, 8, 5` is the mirror. Nothing covers anything at frame 1, so the voters are `run`, `3`, `4`, `8` and
`5`. `run` holds `ANSWER-D` alone and places a `5`; each digit holds both and splits. `5` wins by three shares
to two.

**With the classes formed.** A wider alphabet, or a longer history, makes a class over the digits pay
(D30). `run`'s table then holds two functions: `same`, naming `run`, `P` over slots `1` and `2`, and classes
at `3` and `4`; and `differ`, naming `run` and classes at all four (D38); `equal` has re-centered into
`same` (D29). Both children cover `run`, bind the four digits, and fire at slot `0`, `run`'s (D53), so in
`say`'s table a class `Q` over `same` and `differ` now pays, at slot `0` a frame back, in `from-c` and
`from-d` in place of `EQUAL` and `run`. `Pc` still ties slot `0` now to slot `3` a frame back, where the bound
`c` stands as it always did; `from-c` holds there and is credited the answer, `c` having been credited to
`same` a frame before (R23).

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `run`, `3`, `3`, `8`, `5` | `same` at slot `0`; the four digits bound at slots `1` to `4` | `same` covers `run` and binds the digits. Each infers at its age 0 (R36). |
| 2 | `say` reported as run; nothing reported in `digit` | `ANSWER-C`, over `say` and `same`; the `8` and the expected `8` bound beside it | `same`, covered, connects once to `ANSWER-C` (D10); the four digits, bound, connect to it as before (R31). |

The vote at frame 1, voter by voter: `same` holds `ANSWER-C` alone, since it stands only in equal requests.
Each digit holds both children, from the equal and the unequal requests it has stood in, and splits. So `8`
takes one voter whole and half of four, three shares, and `5` two. The branch voter wins, as in the bare
regime.

# 5. The price

The alphabet is the twelve base neurons here, so a base symbol costs `log₂ 12 ≈ 3.6` bits. `run`'s table holds
one function, so which of them costs nothing, and `P` chooses a member at `log₂ 10 ≈ 3.3` (D13).

| the request, priced as | bits |
|---|---|
| flat: five base symbols | 17.9 |
| `equal`, equal request: `run`, one member for `a` and `b`; `c` and `d` as themselves | 14.1 |
| `equal`, unequal request: `run`, one member, one place failed; `a` not bound | covers 7.2 for 7.9 — not taken |

`say`'s table holds two functions, so which of them costs `1`, and a copy parameter chooses among ten at `3.3`. A
`from-c` call covers `say` and `EQUAL`, at what its own call cost, `6.9`, and binds the digit `c` and the
answer, `17.7` bits in all, for `say`, which function and one choice, `7.9`: it saves about `10` bits on an
equal request in the bare regime, where `c` stood bare and no earlier bid was credited it. `from-d` on an
unequal request covers `run` where `from-c` covered `EQUAL`, and saves about `6.5`. Either saving is positive,
which is all the election asks (R22, R24). Once `same` binds `c` at the request, it is credited there (R23)
and `from-c` is credited only the answer, 3.6 less; it still pays.

# 6. What this case leans on

- **The digits stand on the apex, bare or bound.** No class in `run`'s table pays within the history, so
  the digits stand bare, the copy parameters are grown on them directly, and they vote. Were the alphabet wide
  enough for a class to pay, `same` would bind `c` and `d`, and nothing about the digits would change for
  `say`: a bound digit is the same activation at the same place (D10). The copy is the same, the digits still
  hedge, and the vote at the request is still decided by `EQUAL` or `run`.
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
| the two arms of the conditional | `from-c` and `from-d`, two functions in `say`'s table, taken by which copy parameter holds |
| `return c` | `Pc`, bound from slot `3` and placed beside `say` |
| choosing the arm | no neuron: `EQUAL` or `run` names the child it was followed by, the digits name both, and the share decides (R47) |
| the call site that knows the whole | `ANSWER-C` and `ANSWER-D`, one frame after the request, each with the digits it bound beside it |
