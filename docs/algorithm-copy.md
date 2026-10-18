# Say the digit you were shown, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that is shown a digit, is asked, and says that digit,
written out neuron by neuron. Nothing here is normative, and the machine is shown running: its tables and
connections are as listed, and its histories already hold requests like this one. It shows the smallest copy
the design can make: one parameter over the digit shown and the digit said, bound from what was shown and placed
at what is said (D44, R28), so that a digit never said before is said the first time it is shown. It also shows
the same parameter reading the digit through a class, once the class has paid (D45).

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
| the copy | `W`, a parameter in `say`'s table over two adjacent frames: the digit said is the digit shown (D44); value neurons `W:0` to `W:9`, "said k" |
| the function | `write`, in `say`'s table, naming `say`, what stood a frame before it, and `W` beside it; its child `WROTE` (D48) |
| the shown digit, once it pays | `C`, a class in `run`'s table, members the digits, and `shown`, a function naming `run` and `C` beside it (D41); its child `SHOWN`, and value neurons `C:0` to `C:9`, "shown k" |

# 3. How it was learned

1. **The digits stand bare at first.** `run` fires beside the digit and sees it at offset zero (D26), a
   different one each time. An offset from a thing where neurons vary is what a class is seeded on (D47), and
   with twelve base neurons a class of the ten digits saves `log₂ 12 − log₂ 10`, about a quarter of a bit, each
   time a function names it over one, against a line that names ten neurons (D13, D30). Until the history holds
   enough requests for that to pay, `C` does not form, and the digit and `run` stand bare on the apex.
2. **The copy is a parameter.** `say` fires in the next frame and sees the request's frame at `−1`: `run`, and
   the digit shown; and beside itself at `0`, the digit said. In every taught request the same digit stands at
   `−1` and at `0`: two offsets that hold one neuron is the relation a parameter is seeded on (D47), and its
   collapse makes it `W`, two positions a frame apart (D27). It writes which member once and covers the said
   digit in full. It is open on its members, so a digit never shown before is its value the first time, and a
   member from then on (D44).
3. **The function is grown over it.** `say` stands with `run` a frame before it in every row, and `W` holds in
   every row, so the collapse grows one function over the three, `write` (D33, D27): `W` alone would cover the
   two digits and not `say`, and a function over `W` and nothing else would write what `W` writes alone (D38).
   `say` is the latest frame the bid covers, so its child `WROTE` fires at `say`'s coordinate (D53), covering
   `say`, `run`, the shown digit and the said one, and `W`'s value neuron fires over `W`'s two positions, at
   the said digit's frame, beside it (§7.4).
4. **The request learns its answer.** `run` is covered in the frame the answer runs and connects once to the
   neuron that covered it, `WROTE`, one frame ahead, with that frame's reward (D10). So does the shown digit,
   covered through `W`. Before `write` existed, `say` and the digit stood bare at the answer frame and both
   connected to them directly; those connections stay (R31).
5. **The class forms, and the copy reads through it.** Once the history is long enough, `C` pays (D30), and
   `shown`, the function naming `run` and `C` beside it, with it. From then on `shown`'s bid covers `run` and
   the digit, and what stands on the apex at the request is `SHOWN` and `C:k`, "shown k", both at the request's
   coordinate: the function's child, and its variable with its value (D45). `say` now sees both at `−1`, `C:k`
   carrying the digit as `C`'s value, and `W` holds through it: a digit shown bare and the same digit held in a
   class are one and the same neuron to a parameter (D44). `run` no longer stands, so `write` re-centers (D29):
   `run` at `−1` fails in every use and leaves the line, `SHOWN` at `−1` holds in every use and joins it, and
   `write` is `say`, `SHOWN` a frame before and `W`, with the same child. `WROTE` now covers `say`, `SHOWN`,
   `C:k` and the said digit, and `SHOWN` and `C:k` connect to it one frame ahead (D10).
6. **A digit the class has never admitted.** `C` is closed on its members (D41): a digit shown for the first
   time is no member, `shown` fails, and `run` and the digit stand bare as in step 1. `run` holds `WROTE` from
   the bare lessons, `write` fits with `SHOWN` a failed neighbor, and `W` is open, so the new digit is copied
   the first time it is shown. The class memorizes what recurs; the parameter carries what is new.

# 4. One request, frame by frame, after teaching

**With the digits bare.** The world shows `3` beside `run`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1 | `3`, `run` | `3`, `run` | Each infers at its age 0 (R36). `run` holds `WROTE` one frame ahead; so does the `3`, from every request it was shown in. |
| 2 | `say` reported as run; nothing reported in `digit` | `WROTE`, over `say`, `run`, the `3` and the expected `3` | The expected `3` stands, since nothing replaced it (D40). `run` and the `3`, covered, connect once to `WROTE` (D10). |
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
| 1 | `7`, `run` | `SHOWN` and `C:7`, over `run` and the `7` | Each infers at its age 0: `SHOWN` holds `WROTE` one frame ahead from every request since `shown` formed, and `C:7` holds it from every request a `7` was shown in. |
| 2 | `say` reported as run; nothing reported in `digit` | `WROTE`, over `say`, `SHOWN`, `C:7` and the expected `7` | `SHOWN` and `C:7` connect once to `WROTE` (D10). |

Both inferences to `WROTE` complete at frame 2. Its expansion places `say` there, `SHOWN` at frame 1, where it
stands, and `W` bound from what stands at frame 1, `C:7` read through to the `7`, and placed at frame 2 (R28).
The match proposes it, and `7` is expected and output.

Show a `9` that `C` has never admitted, and the first table runs: `shown` fails, `run` stands bare, `run` infers
`WROTE`, and `W` is bound from the `9`.

# 5. The price

The alphabet is the twelve base neurons here, so a base symbol costs `log₂ 12 ≈ 3.6` bits. `say`'s table holds
two patterns, so which of them costs `1`, and `W` chooses a member at `log₂ 10 ≈ 3.3` (D13).

| the answer, written as | bits |
|---|---|
| flat: `say` and a digit | 7.2 |
| `write`: `say`, which pattern, one member; `run` and the shown digit covered with it | 7.9 for three symbols covered besides `say`, 10.7: saves 2.8 |
| `write` through the class: `say`, which pattern, one member; `SHOWN` and `C:k` covered at what their call cost, `run` and one member, 6.9 | 7.9 for `SHOWN`, `C:k` and the said digit, 10.5: saves 2.6 |

`shown` in `run`'s table covers `run` and the digit for `run`, no pattern choice with one pattern in the table,
and one member: it saves a quarter of a bit a request, and `C`'s line names ten neurons, so it pays only when
the history holds some hundred and fifty requests. That is the one number this case turns on, and it is `H`
(R4).

# 6. What this case leans on

- **One parameter, not ten connections.** A copy by connections would need the request to learn, per digit,
  that saying that digit followed; a digit never shown would have nothing. `W` is seeded on the agreement of
  two positions and holds for any member, so the crossing from what was shown to what is said is learned once.
- **The written digit is an event.** `say` is the only action. What is said is an expected event in the
  digit's own dimension, output beside it (§3.5), so the same neuron stands at the shown and the said
  position, which is what lets a parameter test them (D2, D44).
- **A parameter reads through a class.** Once `C` covers the digit, `W` reads it as `C`'s value and covers
  the value neuron (D44, D45). The copy is the same in both regimes; what changes is who votes: bare, `run`
  and the digit; covered, `SHOWN` and `C:k`.

# 7. The same thing as code

| in the code | in the machine |
|---|---|
| `say(x)` | `say`, with `W` bound to `x` beside it |
| the argument `x` | `W`'s member, read from the request's frame |
| the return value appearing on the screen | the expected digit at the answer frame, which the world reads |
| a value the function has seen before | `C:k`, the class's value neuron, inferring `WROTE` beside `SHOWN` |
| a value it has not | `run` bare, inferring `WROTE`, which binds `W` from what stands |
