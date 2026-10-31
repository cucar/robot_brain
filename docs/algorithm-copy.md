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

**In.** One channel, laid out over time alone, so a base neuron sees one frame back (D1, D4). It declares two
event dimensions, `digit` first, with ten buckets, `0` to `9`, and `run`, with one bucket, `run`. A request is
one frame: the digit in `digit` and `run` beside it.

**Out.** One action dimension in the same channel, `say`, with one base action, `say`. The answer is `say` in
the frame after the request, with a digit beside it in `digit`: the digit is an event the machine infers
there, and both are its output (R30, R40). Two frames of silence follow. An activation is open for its reach
in time (D9), and the interneuron that covers the request from the answer frame reaches two frames (D4), so
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
| the requests | one function per digit that has been shown twice, `{d, run@0}`: the digit and `run` beside it, origin the digit, since `digit` is declared first (D15); each with its interneuron, `RUN3` for the `3` |
| the answers | one function per digit said three times, `{say, d@0, d after d}`: `say`, the digit beside it, and the same digit the frame before, a cycle of one node (D15); each with its interneuron, `SAY3` for the `3` |
| the copy | one function, `write` = `{say, W@0, W after W}`: `say`, a parameter `W` beside it, and the same `W` the frame before, so the digit said is the digit shown (D44, D38); its interneuron `WRITE` |

# 3. How it was learned

1. **The first showing of a digit.** The world shows `3` beside `run`. At frame 1 `run` and the `3` fire, and
   each sees the other at offset zero (D26): one relation at strength 1, and nothing is built on one occurrence
   (R21). At frame 2 `say` and a `3` fire. `say` sees the `3` at `0`, and `run` and the `3` at `−1`; the said
   `3` sees `say` at `0` and the shown `3` at `−1`. All four stand bare, each at its record of `2` (D13). `run`
   and the `3` at frame 1 connect to `say` and to the `3` at `+1`, with the lesson's reward (R31).
2. **The request is named at the second showing.** `(3, run, 0)` has strength 2. Grouping makes `{3, run@0}`
   (R50). A request written bare is `run` and the `3`, `4`; through the function it is the call's record, `2`.
   That saves `2` per request against an entry of `3`, two neurons and one relation: worth `1` at strength 2
   (D57). Find reaches nothing, so the function is new (R53). Elected, it covers `4` for `2` (R24). `RUN3` is
   allocated and fires at frame 1, covering `run` and the `3` (R16, D53). Covered at age 0, both write
   nothing (D10).
3. **The answer is named at the second showing, and closes at the third.** At frame 2 of the second showing
   `say` sees the said `3` at `0`, and `RUN3` at `−1` with nothing under it (D5, D10). `(say, 3, 0)` has
   strength 2 and makes `{say, 3@0}`: `4` for `2`, entry `3`, worth `1`. Elected, `SAY3` fires at frame 2
   covering `say` and the said `3`; `RUN3`, open at age 1, connects to it at `+1` (D10). `(say, RUN3, −1)` is
   at strength 1, since the first showing had `run` and the `3` bare. At the third showing it has strength 2.
   `RUN3`'s origin is the `3`, a node of `{say, 3@0}`, so the relation is read as a relation to that node (R50,
   D54): the `3` a frame back, which is the `3` come again, and the relation that reaches it closes the cycle,
   `3` after `3` (D15). The said `3` answers for it over its history: it saw a `3` one back at every showing,
   bare at the first and as `RUN3`'s origin after, so `with` is 3, `without` 0, and it costs one offset: it
   joins (D57). The answer is `{say, 3@0, 3 after 3}`. Placed again, its fit lands on `say`, the said `3`, and
   the `3` at frame 1 as the call `RUN3` standing there, covering the call's whole count (D54, D22): `6` for
   `3`, the call's record and how far it ran, one lap. Elected; `RUN3` leaves the apex and connects once more,
   to `SAY3` (D10).
4. **A second digit makes the parameter.** The `5` goes the same way: `RUN5`, then `{say, 5@0, 5 after 5}`.
   Find reads each relation as it is and with either end open (R53): `(say, ?, 0)` reaches `SAY3`'s function,
   and the two have the same shape, two nodes and two relations. Paired from their origins (R52), `say` is
   fixed, and the one other node holds `3` in one and `5` in the other: a parameter, `W`. That is `write`,
   `{say, W@0, W after W}`. Its entry is `say`, `W` and two relations, `4`; its call's record is `2`, one value,
   and how far it ran, `4`. Priced as a replacement over the four lessons: the two specifics cost `8` in the
   dictionary and `3` per lesson, `20`; `write` costs `4` and `4` per lesson, `20`. No cheaper, it replaces
   nothing (R52), and it stands beside them for the digits they do not name.
5. **What is written from then on.** On a repeat showing `RUN3` covers the request at frame 1, and at frame 2
   `SAY3` and `write` both fit over `say`, the said `3` and `RUN3`'s call, `6` for `3` and `6` for `4`; the
   election takes `SAY3` (R24). On the first showing of a digit `write` has not seen, `run` and the digit
   stand bare at frame 1: `(9, run, 0)` is at strength 1, its open reading reaches the `{d, run@0}` functions,
   and none fits, since `9` is not the digit any of them names. At frame 2 `write` lands on `say`, the `9` at
   `0`, and the `9` at `−1`, bare: `6` for `4`. `W` takes any neuron: the `9` is its value the first time and a
   member from then on (D44). `run` stands bare at frame 1. The `9`'s second showing gives it `RUN9` as any
   digit's does, and its third `SAY9`.
6. **The class never pays.** Generalizing `{5, run@0}` with `{3, run@0}` differs at one node, which would be a
   class over the digits (R52). Priced as a replacement: `m` functions of one digit each cost `3m` in the
   dictionary and `2` per request; the one function over the class costs `3` and a member each, `3 + m`, and
   `3` per request, one more for the value. The class is cheaper only while the requests number fewer than
   `2 − 3/m` per digit, and each function already has two. So each digit keeps its own function.
7. **The connections.** `RUN_d` is covered by `SAY_d` at every showing from the third on, so it connects once to
   `SAY_d` at `+1` each time (D10). `run` connects to `WRITE` at `+1` at every first showing of a digit, where
   it stood bare. `run`'s connections to `say`, the `3` and the `5` from the first lessons stand at strength `1`,
   and nothing ever leaves a connection (R31). `SAY_d` and `WRITE` connect to nothing: nothing stands on the
   apex within their windows.

# 4. One request, frame by frame, after teaching

**A digit shown before.** The world shows `3` beside `run`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `3`, `run` | `RUN3`, over the `3` and `run` | `RUN3` infers `SAY3` at `+1` (R36). `run` and the `3` are covered and silent (D10). |
| 2 | `say` reported as run; nothing reported in `digit` | `SAY3`, over `say`, the `3` said, and `RUN3` | The inferred `3` stands, since nothing replaced it (D40). `RUN3` connects once more, to `SAY3` (D10). |
| 3, 4 | silence | | |

At frame 1 the inference to `SAY3` completes at frame 2. Its expansion places `say` there and the `3` beside it,
and reads the `3` a frame back as `RUN3`, the call whose origin it is (R28, D54). The match lands on `RUN3`,
credits its record, `2`, and charges nothing: the inference is proposed (R49). `say` is the one candidate in
its dimension and is output; the `3` is the one candidate in `digit` at frame 2 and is output (R47).

**A digit never shown.** The world shows `9` beside `run`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `9`, `run` | `9`, `run` | `run` infers `WRITE` at `+1`, and from the first lessons `say`, the `3` and the `5` at `+1`. The `9` holds no connection. |
| 2 | `say` reported as run; nothing reported in `digit` | `WRITE`, over `say`, the `9` said and the `9` shown; `run` | `run` and the `9` connect once, to `WRITE` (D10). |

`WRITE`'s expansion from `run` places `say` at frame 2 and reads `W` back: `W` a frame before is what stands
at frame 1, the `9`, and `W` is one node, so the `9` is placed beside `say` at frame 2 (R28, D44). The match
lands on the `9` at frame 1, credits its record, `2`, and charges nothing: proposed. The vote in `digit` at
frame 2 has one voter, `run`, its one vote split by strength across the `9` through `write` and the `3` and the
`5` from the first lessons (R47). The connection to `WRITE` has strengthened on every first showing since
`write` formed, so the `9` wins and is output. `say` is the one candidate in its dimension. That is what a
parameter is for.

# 5. The price

The channel is laid out over time alone, so a bare activation's record is its neuron and its frame, `2`
(D13). A call's record is `2`, one per value, one per relation that did not come true, and one for how far a
cycle ran (D12).

| a lesson, priced as | numbers |
|---|---|
| bare: `run`, the digit, `say`, the digit | 8 |
| `RUN_d` over the request, the answer bare: `RUN_d`'s record, `say`, the digit | 6 |
| `SAY_d` over the answer and `RUN_d`'s call: its record and how far it ran | 3 |
| `write` over a bare request, the digit's first showing: its record, `W`'s value, how far; and `run` bare | 6 |

`RUN_d`'s entry is `3`, `SAY_d`'s `4`, and `write`'s `4`. A function over `run` and a class of the digits would
cost `3 + m` for `m` members and `3` per request, against `3m` and `2` per request for the digits' own
functions; it is never cheaper once each digit has two requests (§3). Nothing in this case turns on `H` (R4)
beyond that: a digit's function forms at its second showing within the digit's history.

# 6. What this case leans on

- **One parameter, not ten connections.** A copy by connections would need the request to learn, per
  digit, that saying that digit followed; a digit never shown would have nothing. `W` is made from two
  functions that differ only at the digit (R52) and holds for any neuron, so the crossing from what was shown
  to what is said is learned once.
- **The said digit is an event.** `say` is the only action. What is said is an inferred event in the digit's
  own dimension, output beside it (R30), so the same neuron stands at the shown and the said position, which is
  what lets one node be read at both (D11, D44).
- **The shown digit is reached through the request's call.** Nothing under a call is seen (D5). The digit is
  the request function's origin, so the node for the digit a frame back comes true on `RUN_d`'s call and
  covers it (D54); the copy is the same on a first showing and a repeat, and what changes is who votes: bare,
  `run`; covered, `RUN_d`.
- **The copy is a cycle.** The same digit as the frame before is one node read twice, the second time closing
  the cycle; a parameter is listed at most twice, that way (D38). The fit reads one lap, the frame of the
  request, and stops at the silence before it.

# 7. The same thing as code

| in the code | in the machine |
|---|---|
| `say(x)` | `WRITE`, with the digit bound in `W` beside it |
| the argument `x` | what stands in `W` a frame back, read through the request's call |
| the return value appearing on the screen | the inferred digit at the answer frame, which the world reads |
| a value the function has seen before | `RUN_d`, inferring `SAY_d`, which names the digit |
| a value it has not | `run` bare, inferring `WRITE`, which binds `W` from what stands |
