# Say the digit you were shown, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that is shown a digit, is asked, and says that digit,
written out neuron by neuron. Nothing here is normative, and the machine is shown running: its dictionary and
connections are as listed, and its histories already hold requests like this one. It shows the smallest copy
the design can make: a function with one parameter tying the digit shown to the digit said, bound from
what was shown and placed at what is said (D44, R28), so that a digit never said before is said the first time
it is shown.

A function is written here as its nodes, each at its offset in time from the node in its latest frame. The
dictionary holds it as relations, each written from the older of its two nodes (D15, D34).

---

# 1. The environment

**In.** One channel, laid out over time alone, with radius `1`, so a base neuron sees one frame back (D1).
It declares two event dimensions: `digit`, with ten buckets, `0` to `9`, and `run`, with one bucket, `run`. A
request is one frame: the digit in `digit` and `run` beside it.

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
| the requests | one function per digit that has been shown twice, `{run, d@0}`: `run` and the digit beside it; each with its interneuron, `RUN3` for the `3` |
| the answer | one function, `write` = `{say, run@−1, W@−1, W@0}`: `say`, `run` a frame before it, and a parameter `W` tying the digit's place a frame back to the digit's place beside `say`, so the digit said is the digit shown (D44); its interneuron `WRITE` |

# 3. How it was learned

1. **The first showing of a digit.** The world shows `3` beside `run`. At frame 1 `run` and the `3` fire, and
   each sees the other at offset zero (D26). Each returns that relation held in one row, and nothing is built on
   one occurrence (R21). At frame 2 `say` and a `3` fire. `say` sees the `3` at `0`, and `run` and the `3` at
   `−1`: one row. All four stand bare, each at its record of `2` (D13). `run` and the `3` at frame 1 connect to
   `say` and to the `3` at `+1`, with the lesson's reward (R31).
2. **The request is named at the second showing.** `run` returns `(run, 3, 0)` held in two rows. Grouping
   makes `{run, 3@0}` (R50). A row written bare is `run` and the `3`, `4`; through the function it is the
   call's record, `2`. That saves `4` over two rows against an entry of `3`, two neurons and one relation:
   margin `1` (D30). Find reaches nothing, so the function is new (R53). Elected, it covers `4` for `2` (R24).
   `RUN3` is allocated and fires at frame 1, covering `run` and the `3` (R16, D53). The `3` returned the same
   relation, and the same relations are the same function (D36), so it was placed once. Covered at age 0,
   `run` and the `3` write nothing (D10).
3. **The answer is named at the same showing.** At frame 2 `say` sees the `3` at `0`, `RUN3` at `−1`, and under
   it `run` and the `3` at `−1` (D47). The relations held in two rows are `(say, 3, 0)`, `(say, run, −1)` and
   `(say, 3, −1)`; `RUN3` at `−1` is held in one. Grouping makes `{say, run@−1, 3@−1, 3@0}`. Row 1 is `8`
   bare, `2` through the function. Row 2 is `say`, the `3` at `0`, and `RUN3`'s record, since the function
   lands on both activations under it: `6` for `2` (D22, R23). That saves `10` against an entry of `6`: margin
   `4`. It is new. Elected, it covers `6` for `2`, and takes `RUN3`'s call back: the call leaves the file and
   `RUN3`'s activation leaves the apex (R23). Its interneuron, `SAY3`, fires at frame 2. `RUN3` connects once
   to `SAY3` at `+1`, and so do `run` and the `3` it covered (D10).
4. **A second digit makes the parameter.** The `5` goes the same way: `RUN5` at its second showing, then
   `{say, run@−1, 5@−1, 5@0}` grouped at frame 2. Find reads each relation as it is and with either end open
   (R53). `(say, run, −1)` as it is reaches `SAY3`'s function. `(say, ?, −1)` and `(say, ?, 0)` reach it too.
   It is in every relation's reading, some only open, so the two differ only at the open ends. Generalizing
   (R52): `say` and `run` are the same in both, so they are fixed. The nodes at `−1` and `0` differ, and within
   each function they hold one neuron, `3` in one and `5` in the other, so they are one parameter, `W`. That is
   `write`. Its entry is `say`, `run` and three relations, `5`; its call's record is `2` and one value, `3`. It
   is priced as a replacement over `say`'s four rows: the two specifics cost `12` in the dictionary and `2` per
   row, `20`; `write` costs `5` and `3` per row, `17`. Cheaper by `3`, it joins, and both specifics leave the
   dictionary. Elected at the same frame, it covers `say`, the `5` and `RUN5`'s record, `6` for `3`. `WRITE` is
   allocated and fires at frame 2. `SAY3`'s interneuron is decommissioned once its last window closes (R38),
   and the connections to it go with it (R31).
5. **What is written from then on.** On a repeat showing `RUN3` covers the request at frame 1, and at frame 2
   `write` lands on `say`, the digit, and both activations under `RUN3`, taking the call back: `6` for `3`. The
   relations `say` returns group into nothing new, since `write` fits every group they make, and a function a
   general function fits is not built again (R52, R53). On the first showing of a digit `write` has not seen,
   `run` and the digit stand bare at frame 1: `(run, 9, 0)` is held in one row, its open reading reaches the
   `{run, d@0}` functions, and none fits, since `9` is not the digit any of them names. At frame 2 `write`
   lands on `say`, the `9` at `0`, and `run` and the `9` at `−1`, all bare: `8` for `3`. `W` takes any neuron:
   the `9` is its value the first time and a member from then on (D44). A function for the `9` alone is never
   built. The `9`'s second showing gives it `RUN9` as any digit's does.
6. **The class never pays.** Generalizing `{run, 5@0}` with `{run, 3@0}` differs at one node, which would be a
   class over the digits (R52). Priced as a replacement: `m` functions of one digit each cost `3m` in the
   dictionary and `2` per request; the one function over the class costs `3` and a member each, `3 + m`, and
   `3` per request, one more for the value. The class is cheaper only while the requests number fewer than
   `2 − 3/m` per digit, and each function already has two. So each digit keeps its own function.
7. **The connections.** On every showing since `write` formed, `WRITE` takes `RUN_d`'s call back or covers
   `run` bare, so `run` connects once to `WRITE` at `+1`, the digit shown connects to it the same way, and
   `RUN_d` connects to it when its call is taken back (D10). `run`'s connections to `say`, the `3` and the `5`
   from the first lessons stand at strength `1`, and nothing ever leaves a connection (R31). `WRITE` connects
   to nothing: nothing stands on the apex within its window.

# 4. One request, frame by frame, after teaching

**A digit shown before.** The world shows `3` beside `run`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `3`, `run` | `RUN3`, over `run` and the `3` | `RUN3` infers `WRITE` at `+1` (R36). `run` and the `3` are covered and silent (D10). |
| 2 | `say` reported as run; nothing reported in `digit` | `WRITE`, over `say` and `run`; the `3` shown and the `3` said bound beside it | The inferred `3` stands, since nothing replaced it (D40). `write` takes `RUN3`'s call back; `RUN3`, `run` and the `3` connect once more, to `WRITE` (D10). |
| 3, 4 | silence | | |

At frame 1 the inference to `WRITE` completes at frame 2. Its expansion places `say` there, `run` at frame 1,
and `W` bound from what stands at frame 1, the `3` under `RUN3`, placed at frame 2 (R28, D44). The match lands
on `run` and the `3` under `RUN3`, credits `RUN3`'s record, `2`, and charges nothing: the inference is proposed
(R49). `say` is the one candidate in its dimension and is output; the `3` is the one candidate in `digit` at
frame 2 and is output (R47).

**A digit never shown.** The world shows `9` beside `run`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `9`, `run` | `9`, `run` | `run` infers `WRITE` at `+1`, and from the first lessons `say`, the `3` and the `5` at `+1`. The `9` holds no connection. |
| 2 | `say` reported as run; nothing reported in `digit` | `WRITE`, over `say` and `run`; the `9` shown and the `9` said bound beside it | `run` and the `9` connect once, to `WRITE` (D10). |

`WRITE`'s expansion from `run` places `say` at frame 2, `run` at frame 1, where it stands, and `W` bound from
the `9` at frame 1 and placed at frame 2. The match credits `run`'s record and the `9`'s, `4`, and charges
nothing: proposed. The vote in `digit` at frame 2 has one voter, `run`, its one vote split by strength across
the `9` through `write` and the `3` and the `5` from the first lessons (R47). The connection to `WRITE` has
strengthened on every showing since `write` formed, so the `9` wins and is output. `say` is the one candidate
in its dimension. That is what a parameter is for.

# 5. The price

The channel is laid out over time alone, so a bare activation's record is its neuron and its frame, `2`
(D13). A call's record is `2`, one per value, and one per relation that did not come true (D12).

| a lesson, priced as | numbers |
|---|---|
| bare: `run`, the digit, `say`, the digit | 8 |
| `RUN_d` over the request, the answer bare: `RUN_d`'s record, `say`, the digit | 6 |
| `write` taking `RUN_d`'s call back: its record and `W`'s value | 3 |
| `write` over a bare request, the digit's first showing: the same | 3 |

`RUN_d`'s entry is `3`, and `write`'s is `5`. A function over `run` and a class of the digits would cost
`3 + m` for `m` members and `3` per request, against `3m` and `2` per request for the digits' own functions;
it is never cheaper once each digit has two requests (§3). Nothing in this case turns on `H` (R4) beyond that:
a digit's function forms at its second showing within `run`'s history.

# 6. What this case leans on

- **One parameter, not ten connections.** A copy by connections would need the request to learn, per
  digit, that saying that digit followed; a digit never shown would have nothing. `W` is made from two
  functions that differ only at the digit (R52) and holds for any neuron, so the crossing from what was shown
  to what is said is learned once.
- **The said digit is an event.** `say` is the only action. What is said is an inferred event in the digit's
  own dimension, output beside it (R30), so the same neuron stands at the shown and the said position, which is
  what lets a parameter tie them (D11, D44).
- **A covered digit still stands for itself.** Under `RUN_d` the digit is a neighbor at its own offset
  (D47), so `W` reads it there, and `write` lands on both activations under the call and takes it back (R23).
  The copy is the same on a first showing and a repeat; what changes is who votes: bare, `run`; covered,
  `RUN_d`.

# 7. The same thing as code

| in the code | in the machine |
|---|---|
| `say(x)` | `WRITE`, with the digit bound in `W` beside it |
| the argument `x` | what stands in `W`'s place a frame back |
| the return value appearing on the screen | the inferred digit at the answer frame, which the world reads |
| a value the function has seen before | `RUN_d`, inferring `WRITE`, which binds `W` from the digit under it |
| a value it has not | `run` bare, inferring `WRITE`, which binds `W` from what stands |
