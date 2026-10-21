# Say the digit you were shown, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that is shown a digit, is asked, and says that digit,
written out neuron by neuron. Nothing here is normative, and the machine is shown running: its tables and
connections are as listed, and its histories already hold requests like this one. It shows the smallest copy
the design can make: a function with one parameter tying the digit shown to the digit said, bound from
what was shown and placed at what is said (D44, R28), so that a digit never said before is said the first time
it is shown. It also shows the same variable binding the digit once a class has bound it below.

---

# 1. The environment

**In.** One channel, laid out over time alone, with radius `1`, so a base neuron sees one frame back (D1).
It declares two event dimensions: `digit`, with ten buckets, `0` to `9`, and `run`, with one bucket, `run`. A
request is one frame: the digit in `digit` and `run` beside it.

**Out.** One action dimension in the same channel, `say`, with one base action, `say`. The answer is `say` in
the frame after the request, with a digit beside it in `digit`: the digit is an event the machine expects
there, and both are its output (§3.5, R40). A frame of silence follows, so nothing of one request is within
reach of the next.

**Policy.** Every entry of both tables of the neighborhood policy is on (D50), and the action policy is nothing
(D51): where every candidate in `say` is negative, the machine says nothing.

**Teaching.** While the machine is being taught, the world runs `say` and shows the shown digit beside it in
the frame after each request, and rewards it, every lesson the same (§3.5). Afterwards it runs nothing,
executes what the machine outputs, and pays what was right and what was wrong.

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `run`, and the ten digits |
| base action | `say` |
| the function | `write`, in `say`'s table, naming `say`, `run` a frame before it, and a parameter `W` tying the digit's place a frame back to the digit's place beside `say`: the digit said is the digit shown (D44); its child `WROTE` (D48) |
| the shown digit, once it pays | `shown`, a function in `run`'s table naming `run` and a class `C` beside it, members the digits (D41); its child `SHOWN` |

# 3. How it was learned

1. **The digits stand bare at first.** `run` fires beside the digit and sees it at offset zero (D26), a
   different one each time. An offset from a node where neurons vary is a node-to-open relation (D34, D47), and
   with twelve base neurons a class over the ten digits saves `log₂ 12 − log₂ 10`, about a quarter of a
   bit, each time it binds one, against a class that names ten neurons (D13, D30). Until the history holds
   enough requests for that to pay, `shown` does not form, and the digit and `run` stand bare on the apex.
2. **The copy is a parameter.** `say` fires in the next frame and sees the request's frame at `−1`: `run`,
   and the digit shown; and beside itself at `0`, the digit said. In every taught request the same digit stands
   at `−1` and at `0`: a same-to-same relation (D34, D47), and the pick grows a function from it, `write`, over
   `say` and a parameter `W` tying the two places (D33, D27). The collapse then finds `run` a frame back in
   every use and adds it as a fixed node. `W` costs which member once and covers the said digit in full. It is
   open on its members, so a digit never shown before is its value the first time, and a member from then on
   (D44).
3. **What the call looks like.** `write` names `say`, its owner, so its child `WROTE` fires at `say`'s
   coordinate (D53), covering `say` and `run`. The shown digit and the said digit stand in `W`'s two places,
   bound: priced as one member, and still on the apex (D10, §6.4).
4. **The request learns its answer.** `run` is covered in the frame the answer runs and connects once to the
   neuron that covered it, `WROTE`, one frame ahead, with that frame's reward (D10). The shown digit is bound,
   not covered, so it goes on connecting to `WROTE` every time it is shown (R31). Before `write` existed, `say`
   and the digit stood bare at the answer frame and connected to them directly; those connections stay.
5. **The class forms, and the copy reads through it.** Once the history is long enough, `shown` pays
   (D30): a function in `run`'s table naming `run` and a class `C` beside it, members the digits. From then
   on `shown`'s bid covers `run` and binds the digit, and what stands on the apex at the request is `SHOWN` at
   the request's coordinate with the digit bound beside it. `say` sees both at `−1`. `W`'s place a frame back
   is the digit's own coordinate, which is where the bound digit still stands, so `W` holds exactly as before: a
   digit shown bare and the same digit bound in a class are the same activation. `run` no longer stands on
   the apex, so `write` re-centers (D29): `run` at `−1` fails in every use and leaves, `SHOWN` at `−1` holds in
   every use and joins, and `write` is `say`, `SHOWN` a frame before and `W`, with the same child. `SHOWN`
   connects to `WROTE` one frame ahead (D10), and the digit, bound in both calls, keeps connecting to it (R31).
6. **A digit the class has never admitted.** `C` is closed on its members (D41): a digit shown for the first
   time is no member, `shown` fails, and `run` and the digit stand bare as in step 1. `run` holds `WROTE` from
   the bare lessons, `write` fits with `SHOWN` a failed node, and `W` is open, so the new digit is copied the
   first time it is shown. The class memorizes what recurs; the parameter carries what is new.

# 4. One request, frame by frame, after teaching

**With the digits bare.** The world shows `3` beside `run`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `3`, `run` | `3`, `run` | Each infers at its age 0 (R36). `run` holds `WROTE` one frame ahead; so does the `3`, from every request it was shown in. |
| 2 | `say` reported as run; nothing reported in `digit` | `WROTE`, over `say` and `run`; the `3` and the expected `3` bound beside it | The expected `3` stands, since nothing replaced it (D40). `run`, covered, connects once to `WROTE` (D10); the `3`, bound, connects to it as it always has (R31). |
| 3 | silence | | |

At frame 1 each inference to `WROTE` completes at frame 2. Its expansion places `say` there, `run` at frame 1,
where it stands, and `W` bound from what stands at frame 1, the `3`, and placed at frame 2 (R28, R45). The
match reads the line against frame 1 and proposes it (R49). `say` is the one candidate in its dimension and is
output; `3` is the one candidate in `digit` at frame 2 and is expected, and output (R47).

Show a `9` that has never been shown: nothing differs. `run` infers `WROTE` as before, and `W` is bound from the
`9` at frame 1. That is what a parameter is for.

**With the class formed.** The world shows `7` beside `run`, and `7` is a member of `C`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `7`, `run` | `SHOWN`, over `run`, with the `7` bound beside it | Each infers at its age 0: `SHOWN` holds `WROTE` one frame ahead from every request since `shown` formed, and the `7` holds it from every request it was shown in. |
| 2 | `say` reported as run; nothing reported in `digit` | `WROTE`, over `say` and `SHOWN`; the `7` and the expected `7` bound beside it | `SHOWN` connects once to `WROTE` (D10); the `7` connects to it as before (R31). |

Both inferences to `WROTE` complete at frame 2. Its expansion places `say` there, `SHOWN` at frame 1, where it
stands, and `W` bound from what stands at frame 1, the `7`, and placed at frame 2 (R28). The match proposes
it, and `7` is expected and output.

Show a `9` that `C` has never admitted, and the first table runs: `shown` fails, `run` stands bare, `run` infers
`WROTE`, and `W` is bound from the `9`.

# 5. The price

The alphabet is the twelve base neurons here, so a base symbol costs `log₂ 12 ≈ 3.6` bits. `say`'s table holds
two patterns, so which of them costs `1`, and `W` chooses a member at `log₂ 10 ≈ 3.3` (D13).

| the answer, priced as | bits |
|---|---|
| flat: `say` and a digit | 7.2 |
| `write`: `say`, which function, one member; `run` covered and the two digits bound | 7.9 for three symbols besides `say`, 10.7: saves 2.8 |
| `write` over `SHOWN`: `say`, which function, one member; `SHOWN` covered at what its call cost, `run` and one member, 6.9, and the said digit bound; the shown digit was credited to `shown` already (R23) | 7.9 for `SHOWN` and the said digit, 10.5: saves 2.6 |

`shown` in `run`'s table covers `run` and binds the digit for `run`'s name, no function choice with one function
in the table, and one member: it saves a quarter of a bit a request, and `C` names ten neurons, so it pays only
when the history holds some hundred and fifty requests. That is the one number this case turns on, and it is `H`
(R4).

# 6. What this case leans on

- **One parameter, not ten connections.** A copy by connections would need the request to learn, per
  digit, that saying that digit followed; a digit never shown would have nothing. `W` is grown from the
  agreement of two places and holds for any neuron, so the crossing from what was shown to what is said is
  learned once.
- **The written digit is an event.** `say` is the only action. What is said is an expected event in the
  digit's own dimension, output beside it (§3.5), so the same neuron stands at the shown and the said
  position, which is what lets a parameter test them (D2, D44).
- **A bound digit is still the digit.** Once `shown` binds the digit, it stays on the apex where it stood, so
  `W`'s place a frame back holds the same activation it always did (D10). The copy is the same in both regimes;
  what changes is who else votes: bare, `run`; bound, `SHOWN`.

# 7. The same thing as code

| in the code | in the machine |
|---|---|
| `say(x)` | `WROTE`, with the digit bound in `W` beside it |
| the argument `x` | what stands in `W`'s place a frame back |
| the return value appearing on the screen | the expected digit at the answer frame, which the world reads |
| a value the function has seen before | the digit bound in `C`, inferring `WROTE` beside `SHOWN` |
| a value it has not | `run` bare, inferring `WROTE`, which binds `W` from what stands |
