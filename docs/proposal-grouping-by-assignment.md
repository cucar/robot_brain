# Proposal: grouping by assignment and update (replaces R50)

A change to the spec at 491806c. It replaces "merge, repeatedly" in grouping with one step of Lloyd's algorithm
per activation, run online over the neuron's history. Finding (R53), generalizing (R52), the election and pruning
are unchanged. Variables are left aside: this proposal only concerns functions of fixed nodes.

## Why

R50 merges, for each activation, the two groups whose merge most lowers the cost of writing the neuron's history,
and repeats until no merge lowers it. Three things make that slow and fragile:

- **It is serial.** Each merge changes which merge is best next, so the pairs are priced again after every merge.
- **It starts from nothing at every activation.** A neuron sees nearly the same relations activation after
  activation, and grouping rebuilds its functions from single relations each time, mostly to conclude that the
  function it already has is the one.
- **A function, once made, never changes.** If it is slightly wrong (one relation too many, one too few), the only
  remedy is that it loses on margin, is pruned, and grouping builds a replacement from nothing.

Measured on the implementation at 491806c, grouping was about half of all the work, and the part that grew with
the run: on 200 MNIST images, 15,155 merges were priced, and pricing them took 1.1 million re-writes of history
rows.

The cost model already has the shape of a clustering problem, and Lloyd's algorithm is the efficient way to solve
that shape:

| Lloyd's algorithm | the machine |
|---|---|
| a point | a row of a neuron's history: an activation and the relations that held around it |
| a centre | a function: a set of relations |
| the distance from a point to a centre | what the function saves writing the row (D22) |
| assign each point to its nearest centres | the greedy cover of the row by the dictionary (D28) |
| move each centre to fit its points | **missing**: functions never change |
| how many centres | not fixed: a new function opens where it pays its entry, an unused one is pruned (R18) |

The first half, assignment, is already in the machine: it is the greedy cover that prices a row (D28, D30) and
that the election runs on the board. The proposal adds the second half, the update, and replaces merging with it.

## The rule

> **R50 — Grouping, by assignment and update.** For each activation, the machine reads the relations its neuron
> returned in two or more rows (R21), against the rows of the neuron's history that hold any of them. It then does
> three things.
>
> **Assign.** The greedy cover writes the new row with every function the neuron reads (D28, D31). What the cover
> leaves of a row, of its activation and of each neighbor, is that row's assignment. Every other row's assignment is
> what its cover left when that row was last priced (D30), or the row as it is where nothing covered it.
>
> **Update.** Each function the cover took, anchored at this neuron, all of whose relations the row holds, is
> moved: one relation added from the activation's relations, or one dropped, the move that most raises the
> function's value over the history, repeated while one raises it. The moved function replaces the function where
> writing the history with it in place of the old one is cheaper (D30). It keeps its identity and its interneuron:
> a centre moves; it is not replaced.
>
> **Seed.** The relations whose neighbors the cover left unwritten open one new function, trimmed the same way,
> by adding and dropping one relation at a time, but valued against what each row's assignment left. Where its rows
> number two or more and its margin is positive, it goes to find (R53).
>
> The work is bounded by the activation's relations and by `H`. Nothing waits on a previous merge.

**The value of a function over the history**, as the update and the seed read it, is a sum over the rows that hold
any of its relations. In each such row the function is credited the activation and each neighbor that one of its
relations answers. A neighbor answered in several dimensions is credited once. The update credits each at what it
costs in the row; the seed credits each at what the row's assignment left. From that the row deducts:

- the function's price, as it lands on the new row (D22);
- one for each of its relations the row does not hold.

A row where the result is not positive adds nothing. The function's dictionary entry is subtracted from the
total. This value only chooses the move; whether a move or a seed stands is decided by the margin (D30), priced in
full.

**Assignments are what pruning computes.** Pricing a margin (D30) covers every kind of row of a history with
the dictionary's functions. What each cover leaves is kept per row, and grouping reads it. It is a cache, not
state: it can be stale by the levels since the row was last priced, and the margin that decides is priced in full.
Per §6.6, what is cached, and how, is the implementation's.

## The cost table (§6.6)

| Step | Work |
|---|---|
| group | per activation: the cover of the new row; per function the cover took, and for the seed, `G` moves per step, each one landing on the new row and a sum over the rows holding the relations; one margin per moved function and per seed |

No step is a merge of two groups, and none depends on the merge before it.

## What it changes

- A function now changes over its life. This is the update. Its interneuron and connections stay with it, so what
  the interneuron stands for moves with its function.
- Grouping no longer builds every function from single relations: it builds at most one new function per
  activation, the seed, from what the dictionary left unwritten.
- The update moves a function only from activations of its anchor, the fixed node whose history it was found
  over. A function another neuron can read (D31) is moved when its anchor fires. Its anchor fires in the same
  frame, since every node of a function stands at or before its latest frame.

## Results on the implementation

Measured on the same runs as the build before it (491806c, anchor-free functions).

| Run | Before | After |
|---|---|---|
| Six-bar cycle, 12 settings (H 64/256/1024, R 2/3/6/12), 150 laps | all 60/60 on the last 10 laps | all 60/60 |
| Six-bar cycle, H256 R6, 500 laps | 99.4%, settles on 6 functions | 99.4%, settles on 6 functions: a ring of six, each "this bar after the call before", the last pointing back to the first |
| MNIST, 200 images (7×7), grouping time | 2.6 s | 0.5 s |
| MNIST, 200 images, train / frozen test | 34/200, 21/100 | 34/200, 19/100 |
| Abramov, pass 1: grouping time, accuracy | 1.6 s, 15.9% | 0.11 s, 18.0% |
| Abramov, pass 2: grouping time, accuracy | 7.5 s, 21.1% | 0.27 s, 20.8% |
| test.txt, passes 4 / 5 | 61.9% / 61.1% | 65.1% / 57.1% |

Grouping is 5 to 27 times faster. Pruning is now the larger cost: 11.4 of the 13 seconds of Abramov's second pass.

## Open questions

1. **Long histories with long reach keep extra functions.** At H1024 R12 the cycle predicts perfectly but still
   holds 32 functions after 500 laps, where merging settled on 6. The extras are functions anchored on high
   interneurons whose relations reach a lap or two back, and their margins over a long history are positive. At
   H64 R12 it settles on 6, but only after allocating 187 interneurons, where merging allocated 29. Whether the
   seed should be held to more than two rows, or pruning should weigh recent rows more, is open.
2. **The value ignores taking a call back (R23).** A neighbor inside a call is credited nothing, so a seed prefers
   the call itself to everything it covers. On the cycle this is what makes the ring. Where taking a call back is
   the right function, only the margin can find it.
3. **Variables.** In this picture a variable shows up in the update: a relation whose offset holds across a
   function's rows while the neuron at its far end varies. Generalizing could be done there, as an update move,
   rather than by pairing functions.
