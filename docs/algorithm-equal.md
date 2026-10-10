# If equal, the third, else the fourth, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that is shown four digits `a, b, c, d`, is asked, and
writes `c` where `a` and `b` are the same digit and `d` where they are not — `f(a, b, c, d) = a == b ? c : d` —
written out neuron by neuron. Nothing here is normative, and the machine is shown running: its tables and
connections are as listed, and its histories already hold requests like this one. It shows the one equality
test the design has, which is a parameter (D44); a branch that is two children of one table; and an answer that
no neuron at the request holds, because the digit to write and the branch it depends on first stand together
one frame later, in the step that wrote it. The request answers by a vote: the voters that know the branch name
the digit it takes, the voters that know a digit name their own, and there are more of the former.

---

# 1. The environment

**In.** One channel, laid out over time alone, with radius `R = 4` (D1). It declares one event dimension,
`digit`, with ten buckets, `0` to `9`, and one event dimension, `cue`, with one bucket, `ask`. A request is five
frames: `a`, `b`, `c` and `d` one per frame in `digit`, then `ask` alone. Requests follow one another with the
answer's frame between them: the answer runs in a frame of its own, and the next request's `a` comes after it.

**Out.** One action dimension, `out`, with ten base actions, `out-0` to `out-9`, in the same channel, so an
action is a neighbor of the events it ran beside (D1). What runs there in the frame after `ask` is the answer.

**Teaching.** While the machine is being taught, the world runs the right `out` itself in the frame after each
`ask`, and rewards it, every lesson the same (§3.5). It mixes equal and unequal requests and varies every digit.
Afterwards it runs nothing, executes what the machine outputs, and pays what was right and what was wrong.

**Why `a` and `b` are two frames of one dimension.** The only thing in the design that tests two places for the
same content is a parameter, and what it tests is that one neuron stands at both (D44). A neuron is a
dimension–bucket pair (D2), so `3` in a dimension `a` and `3` in a dimension `b` are two neurons that have
nothing in common, and no pattern could say they are equal short of a function per digit naming both outright,
which is the diagonal memorized and not equality learned. Two positions of one dimension, in time as here or
in space as in [algorithm-hit.md](algorithm-hit.md), is where a parameter holds.

# 2. The neurons

| Kind | Neurons |
|---|---|
| base events | `ask`, and the ten digits |
| base actions | `out-0` to `out-9` |
| the test | `P`, a parameter in `ask`'s table over `(−4, −3)`: `a` and `b` are one and the same digit (D44); value neurons `P:0` to `P:9` |
| the digits, as variables | `A`, `B`, `C`, `D`, classes in `ask`'s table, one position each at `−4`, `−3`, `−2`, `−1`, members the ten digits (D41); value neurons `C:0` to `C:9`, `D:0` to `D:9`, and so on |
| the branch | two functions in `ask`'s table: `same`, naming `P`, `C` and `D`; `differ`, naming `A`, `B`, `C` and `D`; their children `same` and `differ` |
| the request, seen from the answer | `asked`, a class in the table of each `out-k`, one position at `−1`, members `same` and `differ`; value neurons `asked:same` and `asked:differ`, one pair per `out-k` |
| the steps | two functions in the table of each `out-k`: `from-c`, naming `asked` and `C:k` a frame before; `from-d`, naming `asked` and `D:k`; their children `k-from-c` and `k-from-d` |
| the branch, bound to the step | a function in `k-from-c`'s table naming `asked:same` beside it, child `k-if-same`; one in `k-from-d`'s table naming `asked:differ`, child `k-if-differ` |

# 3. How it was learned

1. **The test is a parameter.** `ask` sees the four digits at `−4` through `−1`, each at its own exact offset,
   which is what the radius buys (D6). In every request where `a == b`, one and the same neuron stands at `−4`
   and `−3`: a pair of offsets that agree is the relation a parameter is seeded on (D47), and its collapse makes
   it `P` (D27). Agreement is its whole test, so a digit never seen equal before is `P`'s value the first time
   it is and a member from then on. This is the alternation's `P` with its offsets moved
   ([algorithm-xy.md](algorithm-xy.md)).
2. **The digits are classes.** Every offset holds a different digit from one request to the next, and an offset
   that varies is what a class is seeded on. Each class keeps one position: adding a second, independent digit
   would widen the choice by as much as it covers, so the position test refuses it (D27). Where `P` holds, `−4`
   and `−3` are `P`'s, and the classes `A` and `B` hold only where it does not.
3. **The branch is two functions.** Over the rows where `P` holds, `C` and `D` hold in every one, so the collapse
   grows `same` to name all three. Over the rows where `A` and `B` hold, `C` and `D` hold too, and `differ`
   names the four. Both fit an equal request, since the classes admit any digit, but `same` writes `a` and `b`
   as one member where `differ` writes two, so it covers the same five neurons for less and is taken (D28). In
   an unequal request `P`'s two positions disagree: its value is the nearer digit and the farther is a failed
   neighbor (D44), the bid no longer covers more than it costs, and `differ` is taken instead. Two children,
   one per branch, and no neuron ever says which digits they held: that is in the value neurons beside them.
4. **The answer gets its step.** In the taught frames `out-c` ran one frame after `same` and `out-d` one frame
   after `differ`, each with its reward, and everything on the apex at `ask`'s frame, open at age 1, connected
   to what ran (R31). In `out-7`'s history the frame before holds `same` or `differ` in every row, never both,
   and `C:7` beside `same` or `D:7` beside `differ`. Variables come first (D33): `asked` is the class of the two
   children, and it claims that position, so no function of the table names `same` or `differ` outright (D27).
   `from-c` is grown from `asked` standing with `C:7`, and `from-d` from `asked` with `D:7`. Each is a step:
   inferred, it expands to its bidder, `out-7`, which is output (R28, R30). From then on `same` and `C:7` are
   covered in the frame `out-7` runs, and connect once more, to `7-from-c`, the neuron that covered them (D10).
5. **The branch is bound to the step one level up.** `7-from-c` fires beside `asked:same` in every row but the
   coincidences where `c` and `d` were both `7`, so its table holds a function naming that value neuron at
   offset zero, and its child `7-if-same` covers both (D45, §7.1). It is the one neuron in the machine that
   stands for "wrote 7, from `c`, because `a` equaled `b`", and it stands at the frame of the answer, not the
   frame of the request. Whatever is open and uncovered at the request connects to it there.

Nothing at the request's frame ever holds that conjunction. `C:7` is "7 was shown third", whether or not the
pair was equal; `same` is "the pair was equal", whatever was shown; and a level that put them together as one
neuron there would cover what it writes and save nothing, since a random `c` is incompressible (D13). The
machine answers without it, as the next section shows.

# 4. One request, frame by frame, after teaching

`3, 3, 8, 5`. The answer is `8`.

| Frame | input | on the apex | what happens |
|---|---|---|---|
| 1–4 | `3`, `3`, `8`, `5` | the digits | Each stands bare: nothing in a digit's table pays for the digits around it (§6). |
| 5 | `ask` | `same`, `P:3`, `C:8`, `D:5` | `same` covers `ask` and the four digits. The four vote on frame 6 (R47); `out-8` takes about three voters' shares and `out-5` most of one. |
| 6 | `out-8` reported as run | `8-if-same`, over `8-from-c` and `asked:same`, over `out-8`, `same`, `C:8`; `P:3` and `D:5` still | Everything open and uncovered connects to `8-if-same` with the reward; `same` and `C:8`, covered, connect to `8-from-c` (D10). |

**The vote at frame 5**, voter by voter. Each reads its connections at offset one and expands what it finds
(R28); an inference that places in frame 5 a neighbor that did not stand, or a variable no member of which
stood, is struck (R49).

| Voter | its connections at offset one, from the taught rows | what each places at frame 5 | proposes |
|---|---|---|---|
| `same` | `k-from-c` for every `k`: the neuron that covered it each time | `C:k`, and `asked`; only `C:8` stands | `out-8` |
| `P:3` | `k-if-same` for every `k`, from the equal requests of threes it stood uncovered in | `C:k` and `same`; only `C:8` stands | `out-8` |
| `C:8` | `8-from-c`, its coverer in every equal request with `c = 8`; `k-if-differ` for assorted `k`, from unequal requests with `c = 8` | `8-from-c` places `C:8` and `asked`, which `same` satisfies; `k-if-differ` places `differ`, which did not stand | `out-8` |
| `D:5` | `5-from-d`, its coverer in every unequal request with `d = 5`; `k-if-same` for assorted `k`, from equal requests with `d = 5` | `5-from-d` places `D:5` and `asked`, which `same` satisfies too; `k-if-same` places `C:k` and `same`, which only `k = 8` fits | `out-5`, by about ten to one over `out-8` |

Every estimate is the teacher's reward, so the candidates tie on estimate and the larger share of voters wins
(R47): `out-8`, by about three to one. The voter that would have been wrong is not silenced; it is outvoted.
`D:5` knows as much about `5` as `C:8` knows about `8`, but its own step holds the branch only as the class
`asked`, which either branch satisfies, so it names its digit whatever the pair was. `same` names nothing
but the digit `c` held, because every step it is connected to names `C:k` outright, and `P:3` the same. What
decides is that the request fires two voters that know the branch beside one that knows the wrong digit.

Each voter also keeps what it wrote before the steps and the level above existed: connections to `out-k`
outright, for every `k` the early lessons ran, since nothing leaves a connection (R31). Those still place their
digits and are not struck, a base action placing nothing in the past. They are spread over the ten digits and
thin beside what came after, so they hedge every voter a little and move no result here.

`3, 4, 8, 5` is the mirror. `differ` fires with `A:3`, `B:4`, `C:8`, `D:5`: `differ` and the two classes name
`D:k` outright or are bound to `differ`, so they propose `out-5`; `C:8` proposes `out-8` through `8-from-c`;
`out-5` wins by about four to one.

**Before the top level forms.** Until `k-if-same` exists, a voter uncovered at the answer connects to
`k-from-c` and `asked:same` as two apex activations. `k-from-c` still places `C:k` and only the right one
stands; `asked:same` alone expands to `out-k` after `same`, which is true of every `k` in an equal request, so
`P:3` and the minority of `D:5` hedge over the digits. `out-8` still wins, by about two to one, on `same` and
`C:8`. The function of step 5 is what takes the hedge away.

# 5. The price

Say the alphabet is the twenty-one base neurons here, so a base symbol costs `log₂ 21 ≈ 4.4` bits, and `ask`'s
table holds seven patterns, so which of them costs `2.8`. A class of ten chooses a member at `3.3`, and so does
`P` (D13).

| `ask` and four digits, written as | bits |
|---|---|
| flat: five base symbols | 22.0 |
| `differ`: `ask`, which pattern, four members | 20.5 |
| `same`: `ask`, which pattern, three members | 17.2 |
| `same` in an unequal request: as above, one position failed, `a` not covered | covers 17.6 for 19.2 — not taken |

A class of every digit saves only what the alphabet is wider than the digits, `4.4 − 3.3` a digit, and pays
because the bid covers its bidder too; the parameter saves a whole digit on top, which is what makes the two
branches cost differently. In `out-7`'s table `asked` writes a child that cost `7.2` as one bit, and
`from-c` covers `out-7`, `same` and `C:7`, `14.9` bits, for `out-7`, which pattern and one bit, about `7`. The
function above it covers one bit a use and names one neuron, so it waits for the uses to pay the line, and is
the last thing to form.

# 6. What this case leans on

- **A majority with tied estimates.** The branch is read at the request by nothing but the vote, and the vote
  is right because the branch's voters outnumber the wrong digit's. Estimates do not separate them while the
  teacher pays every lesson the same; they tie, and the share decides (R47). A teacher whose rewards varied in
  size would hand the choice to noise in the estimates, since nothing weighs exposures. What pulls the wrong
  step down afterwards is the machine's own errors: `out-5` run in an equal request is paid negatively, `D:5`
  connects to `5-from-d` with that, and the mean falls (R31, R32). This is the route
  [algorithm-evaluation.md](algorithm-evaluation.md) marks as reasoned on the mixed addition case and not run,
  and with one more thing in it: the digit's voter connects to the step that covered it (D10), whose line holds
  the branch as a variable, so the branch cannot strike it; only the level above, `k-if-same`, carries the
  branch outright, and only the uncovered voters reach that.
- **The digits must stand flat for `ask`.** A digit sees the digits before it as `ask` does, and a class in a
  digit's table of whatever stood one back, as a letter holds in [algorithm-text.md](algorithm-text.md), would
  cover `a` with `b` and `c` with `d` before `ask` fired, as a value neuron per ordered pair; `P` cannot hold over
  those. It is held off by price alone: such a class saves `4.4 − 3.3` a use, its line names ten neurons, and a
  digit sees an action one back a quarter of the time, which keeps the action's buckets out of its members and
  the saving at that. `ask`'s own classes save the same per use and pay first only because `ask` sees nothing
  else. `H` sets how soon each pays (R4), and it is the one thing to watch when this is run.
- **Teaching is mixed.** Taught equal requests first, `ask` would hold ten functions naming `a`, `b` outright
  before `P` had two rows to be seeded on; `P` forms from the rows the functions fail, as the general form does
  in [algorithm-xy.md](algorithm-xy.md), and nothing is lost but time.

# 7. The same thing as code

| in the code | in the machine |
|---|---|
| `a == b` | `P` holding at `−4` and `−3`: one neuron at both |
| the two arms of the conditional | `same` and `differ`, two children of `ask`'s table |
| `c` and `d` as values | `C:k` and `D:k`, value neurons beside whichever child fired |
| `return c` | `from-c`, in each `out-k`'s table: the step that wrote `k` from `C:k` |
| choosing the arm | no neuron: the arm's child and its value neurons name the digit the arm takes, the other digit's voter names its own, and the arm has the votes (R47) |
| the call site that knows the whole | `k-if-same` and `k-if-differ`, one frame after the answer, where the branch and the digit first stand together |
