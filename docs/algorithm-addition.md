# Addition, as a machine that could be built

A worked case for [algorithm.md](algorithm.md): a machine that adds two binary numbers, written out neuron by
neuron. Nothing here is normative, and the machine is shown running: its dictionary and connections are as
listed, and its histories already hold sums like this one. It shows the parts of the design doing one small
job: a request with each column; a neuron for each case that can be asked; a function per case that writes
the answer; and a carry that the world shows while it teaches and the machine infers afterwards.

A function is written here as its nodes, each at its offset in time from the node in its latest frame. The
dictionary holds it as relations, each written from the older of its two nodes (D15, D34).

---

# 1. The environment

**In.** Each frame the world presents one column of the sum, the two numbers read from the right: the event
`add`, which is the request, the digit of the first number in event dimension `a`, and the digit of the second
in event dimension `b`. Each digit dimension has two buckets, `0` and `1`. Every dimension is laid out over
time alone, with radius `R = 1` (D1): a neuron at the base sees the frame before it, which is all this case
needs. When the numbers run out the world fires `add` once more with the event `end` beside it in place of
digits, asking for the last digit.

**Out.** One action dimension, `out`, with two base actions, `out-0` and `out-1`. What runs there, in the frame
after a request, is the next digit of the result, from the right.

**Teaching.** While the machine is being taught, the world runs the right digit itself in the frame after each
request, and rewards it (R29). Where the column carried, the world also shows the event `carry` in that frame.
Where a carry comes in, it shows `carry` beside the request. It teaches one case at a time: the same column,
without a carry coming in and then with one, over and over, before it moves to the next. A lesson is a request
frame, an answer frame, and two frames of silence, so nothing of one lesson is open at the next (D9).
Afterwards the world runs nothing and shows no carry, and the columns come one frame after another.

# 2. The carry

`carry` is an event. In training the world shows it. After training nothing reports it: the function that
answers a carrying column is expanded, and the expansion places `carry` in the frame the answer runs in,
weakly, as an inferred event (R30). The world reports nothing in its place, so the inference is what the
machine has in that frame, and a weak activation is an input of its frame like a strong one (D40). That frame
is also the frame of the next request, so `add` sees `carry` beside the next column's digits. Writing the
carry is inferring it; reading it is seeing it beside the next request.

# 3. The neurons

| Kind | Neurons |
|---|---|
| base events | `add`, `a0`, `a1`, `b0`, `b1`, `end`, `carry` |
| base actions | `out-0`, `out-1` |
| the cases | ten functions, each naming the request and what stands beside it at offset zero: the two digits or `end`, and `carry` where one came in; each with an interneuron, `C11` for `1, 1` and `C11c` for `1, 1, carry` |
| the answers | ten functions, each naming the digit that ran, `carry` beside it where the column carries, and a frame before it everything its case names; each with an interneuron, `A11` for the answer to `1, 1` |

# 4. How it was learned

1. **The first lesson of a case stands bare.** `add`, `a1` and `b1` fire at frame 1, each seeing the other two
   at offset zero (D26), one row each; nothing is built on one occurrence (R21). At frame 2 `out-0` runs and
   `carry` is shown. The three at frame 1 connect to both at `+1`, with the reward (R31).
2. **A request and its column become one neuron.** At the second lesson `add` returns `(add, a1, 0)` and
   `(add, b1, 0)`, each at strength 2, and grouping makes `{add, a1@0, b1@0}` (R50). A request written bare is
   `6`; through the function it is the call's record, `2`. That saves `4` per lesson against an entry of `5`,
   three neurons and two relations: worth `3` at strength 2 (D57). `a1` and `b1` return the same relations, and the
   same relations are the same function (D36). It is new (R53). Elected, it covers `6` for `2` (R24), and its
   interneuron, `C11`, fires at frame 1 covering all three (D53). Eight columns give eight such functions, and
   the last request gives two more: `end` with a carry, and `end` without.
3. **The answer gets its function at the next lesson.** At frame 2 `out-0` sees `carry` at `0` and `C11` at
   `−1`, and nothing under it (D5, D10). `(out-0, carry, 0)` is at strength 2 and `(out-0, C11, −1)` at 1,
   since the first lesson had the three bare. At the third lesson both are at strength 2, and grouping makes
   `{out-0, carry@0, C11@−1}`. A lesson's answer frame written bare is `out-0`, `carry` and `C11`'s record,
   `6`; through the function it is the call's record, `2`, the call landing on `C11` as a node and covering
   its count (D22, D54). Savings `4` per lesson, entry `5`, three neurons and two relations, worth `3` at
   strength 2 (D57). Elected, it covers `6` for `2`; `C11`'s activation leaves the apex (D10). `A11` fires at
   frame 2.
4. **The case learns its answer.** `C11` is covered in the frame `A11` fires, and connects once, to `A11` at
   `+1`, with that frame's reward (D10); `add`, `a1` and `b1` under it were covered at age 0 and write nothing.
   That is the only connection `C11` ever writes, and it strengthens at every lesson. `A11` connects to
   nothing: nothing stands within its window.
5. **A carry coming in is a fourth neuron.** In the lessons with a carry, `carry` stands beside the request,
   `C11` covers the other three at `2`, and `carry` stands bare at `2`. `{add, a1@0, b1@0, carry@0}` writes
   such a request at `2` for `8`: it saves `2` over `C11` per lesson, against an entry of `7`, so its worth is
   positive from the fourth carry lesson. Find reaches `C11` through `(add, a1, 0)` as it is and `(add, ?, 0)`
   open; the two do not differ only in which neuron stands at some of their nodes, since one has a node the
   other lacks, so generalizing makes nothing (R52). The function is new, and `C11c` is allocated. Where
   `carry` stands it covers `8` for `2` and is taken before `C11`, which covers `6` for `2` (D28). Where it
   does not, `carry` is a relation that did not come true: it covers `6` for `3`, and `C11` is taken. The
   answer, `{out-1, carry@0, C11c@−1}`, forms once `C11c` has stood before it twice: `out-1`, `carry` and
   `C11c`'s count, `6` for `2` at each lesson, entry `5`, worth `3`. `C11c` connects once to its interneuron,
   `A11c`, at `+1`.
6. **Classes never pay.** `{add, a0@0, b1@0}` is found through `C11`'s open end at `a1`, and generalizing
   the two would put a class over `a0` and `a1` at that node (R52). Priced as a replacement: the two specifics
   cost `10` in the dictionary and `2` per lesson; the one function over the class costs `7`, `add`, `b1`, the
   class, two relations and two members, and `3` per lesson. Over the four lessons the two came from that is
   `18` against `19`, and the gap widens with every lesson. The same holds at every digit's place, so each
   case keeps its own function.

| Case, beside `add` | its answer | names beside the digit |
|---|---|---|
| `0`, `0` | `out-0` | |
| `0`, `1` | `out-1` | |
| `1`, `0` | `out-1` | |
| `1`, `1` | `out-0` | `carry` |
| `0`, `0`, `carry` | `out-1` | |
| `0`, `1`, `carry` | `out-0` | `carry` |
| `1`, `0`, `carry` | `out-0` | `carry` |
| `1`, `1`, `carry` | `out-1` | `carry` |
| `end` | `out-0` | |
| `end`, `carry` | `out-1` | |

Each case's interneuron infers its answer's interneuron at `+1` (R36). Expanded, the answer places the digit in
`out` and `carry` beside it where the function names it, and the case's call a frame back, and through it the
case's neurons (R28). The match lands on the case's call, credits its record and charges nothing, so the
inference is proposed (R49).

The last request always has something beside it to name, `end`, so the end of the numbers is a situation like
any column. Without a carry its answer is a leading `0`.

# 5. One sum, frame by frame, after teaching

`101 + 111`. The columns, from the right, are `(1, 1)`, `(0, 1)`, `(1, 1)`. What is chosen in one frame runs in
the next (R29).

| Frame | the world shows | runs, and is inferred | on the apex | infers |
|---|---|---|---|---|
| 1 | `add`, `1`, `1` | | `C11` | `A11`: `out-0` and `carry` at frame 2 |
| 2 | `add`, `0`, `1` | `out-0`; `carry` inferred | `C01c`, `A11` | `A01c`: `out-0` and `carry` at frame 3 |
| 3 | `add`, `1`, `1` | `out-0`; `carry` inferred | `C11c`, `A01c` | `A11c`: `out-1` and `carry` at frame 4 |
| 4 | `add`, `end` | `out-1`; `carry` inferred | `Cendc`, `A11c` | `Aendc`: `out-1` at frame 5 |
| 5 | | `out-1` | `Aendc` | nothing |

At frames 2, 3 and 4 the inferred `carry` is named twice: by the answer function, as what the column carried,
and by the next case, as the carry coming in. The election credits it to the case, whose coverage over price is
higher, `8` for `2`, and takes the answer function after it for the digit and the case's call it covers, `4`
for `2` (R22, R24). The case's interneuron is strong, since the digits it covers are (D40). The answer dimension
has one candidate each frame, placed at the estimate of the case's one connection, and it is output (R47).

The digits that ran in `out`, from the right: `0`, `0`, `1`, `1`. The result is `1100`, a sum the machine need
never have seen: every request is one of the ten cases, and the carry takes each column's outcome into the
next.

# 6. Taught in another order

Had the columns been mixed from the first lesson, the same ten functions would form, each at the second
lesson of its case, since the digits' places never pay as classes (§4). A function over `add` and one digit
may form first, where that digit stood beside `add` in two lessons of different cases: it saves `2` per
lesson against an entry of `3`. Once the cases it stood in have their own functions, those cover all three
neurons for `2` and are taken first, and the function over two is placed and declined at every lesson. Its
relation still recurs in `add`'s history, since the three fire bare at the base before anything covers them,
so nothing prunes it: it stays in the dictionary, placed and never taken, the case the evaluation document
names. The dictionary otherwise comes out the same: a neuron per case at the first level.

# 7. The same algorithm as code

Written as code, before this design existed, the algorithm was recursive rather than looping. Each of its
parts has a place here:

| in the code                           | in the machine                                                |
|---------------------------------------|---------------------------------------------------------------|
| calling the function                  | `add`, fired with each column                                 |
| `addDigits(d1, d2)`, a lookup of cases | the ten cases, and the answer each infers                    |
| `getDigit(num, pos)`                  | the world presenting the next column                          |
| the `carry` parameter                 | `carry`, shown while teaching and inferred afterwards         |
| recursion over `pos`                  | the frame: each request is one of the same cases              |
| the base case: no digits left         | `end` beside the request, answered by the last carry or a `0` |
