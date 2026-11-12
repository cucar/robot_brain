# Say the digit you were shown, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that is shown a digit, is asked, and says that digit,
written out neuron by neuron. Nothing here is normative, and the machine is shown running: its dictionary and
connections are as listed, and its histories already hold requests like this one. It shows the smallest copy
the design can make: a function with one parameter, read once beside the answer and once more a frame back,
bound from what was shown and placed at what is said (D44, D38, R28), so that a digit never said before is
said the first time it is shown.

A function is written here as its nodes, each at its offset in time from the origin. The dictionary holds it
as relations (D15, D34).

---

# 1. The environment

**In.** One channel, laid out over time alone, so a base neuron sees one frame back, and its row takes what
followed a frame later (D1, D4, D18). It declares two event dimensions, `digit` first, with ten buckets, `0` to
`9`, and `run`, with one bucket, `run`. A request is one frame: the digit in `digit` and `run` beside it.

**Out.** One action dimension in the same channel, `say`, with one base action, `say`. The answer is `say` in
the frame after the request, with a digit beside it in `digit`: the digit is an event the machine infers
there, and both are its output (R30, R40). Two frames of silence follow. An activation is open for its reach
in time (D9), and the interneuron that covers a lesson from the answer frame reaches two frames (D4), so
nothing of one request is open at the next.

**Policy.** Every entry of both tables of the neighborhood policy is on (D50), and the action policy is nothing
(D51): where every candidate in `say` is negative, the machine says nothing.

**Teaching.** While the machine is being taught, the world runs `say` and shows the shown digit beside it in
the frame after each request, and rewards it, every lesson the same (R29). Afterwards it runs nothing,
executes what the machine outputs, and pays what was right and what was wrong.

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `run`, and the ten digits |
| base action | `say` |
| the requests | one function per digit that has been shown twice, `{d, run@0}`: the digit and `run` beside it, a pair, never placed (R24); no interneuron |
| the copy | one function, `write` = `{say, W@0, run@−1, W after W}`: `say`, a parameter `W` beside it, `run` the frame before, and the same `W` the frame before, a cycle of one node over `W` (D15), so the digit said is the digit shown (D44, D38); its interneuron `WRITE`. It answers for every digit: the two digits' answers it was read from left the dictionary when it was taken (R52) |

# 3. How it was learned

1. **The first showing of a digit.** The world shows `3` beside `run`. At frame 1 `run` and the `3` fire, and
   each sees the other at offset zero (D26): one relation at strength 1, and nothing is built on one occurrence
   (R21). At frame 2 `say` and a `3` fire. `say` sees the `3` at `0`, and `run` and the `3` at `−1`; the said
   `3` sees `say` at `0` and the shown `3` at `−1`. All four stand bare, the lesson's graph three relations,
   since the rest are implied by those (D13, D55). `run` and the `3` at frame 1 connect to `say` and to the `3`
   at `+1`, with the lesson's reward (R31), and their rows take both at `+1` (D18).
2. **The request is a pair, and a pair is never placed.** At the second showing `(3, run, 0)` has strength 2
   and starts a function, `{3, run@0}` (R50). Its fit covers one relation for the call's own relation, one for
   one, and is not taken (R24): `run` and the `3` stand bare at every request, and the function waits in the
   dictionary at a worth of −1 (D57).
3. **The answer is named at the second showing, whole.** At frame 2 of the second showing `(say, 3, 0)` has
   strength 2 and starts `{say, 3@0}`. At its fit, four relations are candidates at strength 2 (R50). The said
   `3` after the shown `3` leads back to a node the function has, closing a cycle, so it goes first (D15, R50):
   the `3` counts it, its row holding `say` at `0` and the `3` at `−1` at both answer frames, `with` 2,
   `without` 0, worth 1. `say` after `run` adds a node: `say` counts it, its row holding the `3` at `0` and `run`
   at `−1`, worth 1. The other two, `say` after the shown `3` and the said `3` after `run`, are implied once
   those hold (D34). Joined one at a time, each raises the function's worth, from −1 to 1, the three relations'
   worths less the two occasions (D57, R54). The answer is `{say, 3@0, run@−1, 3 after 3}`. Placed again
   at this frame, its fit enters at `say`, lands on the said `3`, on `run`, and on the shown `3` as the one lap
   of the cycle, which reaches silence before it and ends without a failure (D54): three relations for the
   call's own, 3 for 1, taken. `SAY3` is allocated and fires at the fit's two ends (R16, D53): its start at the
   answer frame and its end at the shown `3`, the relation between them, `−1`, its own. `run`, between the
   ends, is gone with its relations; the shown `3` is replaced by the end (D10). `run` and the shown `3`
   connect once more, to `SAY3` at `+1` (D10); `say` and the said `3`, covered at age 0, write nothing.
4. **A second digit makes the parameter.** The `5` goes the same way to `{say, 5@0, run@−1, 5 after 5}`. Find
   reads each relation as it is and with either end open (R53): `(say, ?, 0)` reaches `SAY3`'s function, and
   the two have the same shape, three nodes and three relations. Paired from their origins (R52), `say` and
   `run` are fixed, and the one other node, reached twice, holds `3` in one and `5` in the other: a parameter,
   `W`. That is `write`, `{say, W@0, run@−1, W after W}`. It is taken where it saves more than the two answers
   save together (R52, D57). An answer saves its three relations at each lesson of its digit, less its entry
   of `3`. `write` saves three relations less one binding at each lesson of any digit, less its entry of `3`.

   | lessons so far | the answers for `3` and `5` save | `write` saves | |
   |---|---|---|---|
   | `3, 3, 5, 5` | `2·3 − 3` each, `6` together | `4·2 − 3`, `5` | not made, and the answers stay |
   | `3, 7, 3, 5, 5` | `6` together | `5·2 − 3`, `7` | taken, and the answers leave |

   This machine was taught in the second order. The `7` was shown once and has no answer of its own (R21),
   but its lesson holds `write`'s three relations with the `7` at `W`, and the count reads it (D47). So
   `write` joins the dictionary, and the answers for `3` and `5` leave it.
5. **What is written from then on.** At every showing, of a digit shown before or of one never shown, `run`
   and the digit stand bare at frame 1 as at any request. At frame 2 `write` enters at `say` and lands on the
   digit at `0`, `run` at `−1`, and the digit at `−1`: three relations for one binding, taken (R24). `WRITE`
   fires with the digit bound in `W`. `W` takes any neuron: the `9` is its value the first time it is shown
   and a member from then on (D44). At the `9`'s second showing `(say, 9, 0)` is at strength 2 and no function
   holds it, so an answer for the `9` is grouped as the `3`'s was (R50), and taken. It has `write`'s shape,
   and `write` already stands for it, so it leaves the dictionary at once (R52). The `3` and the `5` are
   answered the same way, each showing one binding more than under an answer of its own.
6. **The connections.** `run` and the shown digit are covered by `WRITE` at every showing since `write` was
   taken, so each connects once to `WRITE` at `+1` each time (D10). What they connected to in the lessons
   before `write` was taken stands, `run`'s connections to `say` and to the `3`, the `7` and the `5` from the
   first lessons among them, since nothing ever leaves a connection (R31). `WRITE` connects to nothing:
   nothing stands on the apex within its window.

# 4. One request, frame by frame, after teaching

**A digit shown before.** The world shows `3` beside `run`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `3`, `run` | `3`, `run` | Both infer `WRITE` at `+1` (R36); `run` infers `say`, the `3`, the `7` and the `5` from the first lessons besides. |
| 2 | `say` reported as run; nothing reported in `digit` | `WRITE`, over `say`, the `3` said, `run` and the `3` shown, the `3` bound in `W` | The inferred `3` stands, since nothing replaced it (D40). `run` and the shown `3` connect once more, to `WRITE` (D10). |
| 3, 4 | silence | | |

At frame 1 the inference to `WRITE` completes at frame 2. Its expansion from `run` puts `say` there and reads
`W` back: `W` a frame before is what stands at frame 1, the `3`, and `W` is one node, so the `3` stands beside
`say` at frame 2 (R28, D44). The match lands on `run` and the `3` at frame 1, credits both and charges
nothing: the inference is proposed (R49). `say` is the one candidate in its dimension and is output; in
`digit` the `3` is named by two voters, the shown `3` whole and `run` by its share, and is output (R47).

**A digit never shown.** The world shows `9` beside `run`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `9`, `run` | `9`, `run` | `run` infers `WRITE` at `+1`, and from the first lessons `say`, the `3`, the `7` and the `5` at `+1`. The `9` holds no connection. |
| 2 | `say` reported as run; nothing reported in `digit` | `WRITE`, over `say`, the `9` said, `run` and the `9` shown, the `9` bound in `W` | `run` and the `9` connect once, to `WRITE` (D10). |

The request runs as the `3`'s did, with one voter fewer. `WRITE`'s expansion from `run` puts `say` at frame 2
and reads `W` back: what stands at frame 1 is the `9`, so the `9` stands beside `say` at frame 2 (R28, D44).
The match lands on `run` and the `9` at frame 1, credits both and charges nothing: proposed. The vote in
`digit` at frame 2 has one voter, `run`, its one vote split by strength across the `9` through `write` and
the `3`, the `7` and the `5` from the first lessons (R47). The connection to `WRITE` has strengthened at every
showing since `write` was taken, so the `9` wins and is output. `say` is the one candidate in its dimension. That is
what a parameter is for: nothing in the dictionary names a digit's answer, so a digit shown a hundred times
and a digit never shown are answered by one call, the digit bound beside it.

# 5. The price

A bare activation costs its relations, one relation each, and a call one per relation that did not come true
and one per argument, and nothing of its own (D12, D13).

| a lesson, priced as | relations |
|---|---|
| bare: the four activations, joined by three relations, the rest implied | 3 |
| `write` over the lesson: `W`'s binding | 1 |

`write`'s entry is `3`. The request pair's entry is `1`, its relation implied inside the answer. Nothing in
this case turns on `H` (R4) beyond that: the first two digits' answers form at each one's second showing
within the digit's history, and `write` is taken once it saves more than they do (§3).

# 6. What this case leans on

- **One parameter, not ten connections.** A copy by connections would need the request to learn, per
  digit, that saying that digit followed; a digit never shown would have nothing. `W` is made from two
  functions that differ only at the digit (R52) and holds for any neuron, so the crossing from what was shown
  to what is said is learned once.
- **The said digit is an event.** `say` is the only action. What is said is an inferred event in the digit's
  own dimension, output beside it (R30), so the same neuron stands at the shown and the said position, which is
  what lets one node be read at both (D11, D44).
- **The request has no call.** A pair is never placed, so `run` and the shown digit stand bare at every
  request, and the answer lands on them as themselves, the digit at the call's end and `run` between the ends
  (D10). What infers the answer is the two of them, each connected to `WRITE`, and on a first showing `run`
  alone.
- **The copy is a cycle.** The same digit as the frame before is one node read twice, the second time closing
  the cycle (D38). The fit reads one lap, the frame of the request, and stops at the silence before it; the
  relation between the call's ends says so.

# 7. The same thing as code

| in the code | in the machine |
|---|---|
| `say(x)` | `WRITE`, with the digit bound in `W` beside it |
| the argument `x` | what stands in `W` a frame back, read as it stands at the request |
| the return value appearing on the screen | the inferred digit at the answer frame, which the world reads |
| a value the function has seen before, or has not | the same call: `run` inferring `WRITE`, which binds `W` from what stands |
