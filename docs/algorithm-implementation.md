# Algorithm Implementation

The plan for building [algorithm.md](algorithm.md) on the code structure of `brain/brain-core`. A prototype that
implements the spec exists on the branch `origin/ucar`, under `brain/brain-core`. Its logic is right and its
structure is not this project's. The port carries over two things: the logic of that code, and its benchmarks. It
is a logical port, read with `git show origin/ucar:<path>` or from an export into a scratch folder; nothing is merged
or checked out, and nothing under `docs` comes from `ucar`.

The port is done when every benchmark in `origin/ucar:brain/brain-core/benchmarks` gives the same results on the
ported code: the same right counts, the same functions, the same last change of the dictionary, the same profit.
Two rules of the spec at `a30fb88` go on top after that, once the port is verified (section 6).

## 1. The map

The structure in `brain/brain-core` stays: the `Brain` drives the frame, the thalamus owns the regions, a region
its columns, a column its neurons, and every call that reaches neurons is a batch the thalamus fans out over the
columns in parallel. The prototype's parts land in it as follows.

| prototype (`origin/ucar`) | what it is | where it lives in the port |
|---|---|---|
| `machine.rs`: `Machine`, `process_frame`, `frame_up_levels`, `run_half`, `frozen_half`, `start_one_at_a_time`, `activate_level`, `board_span` | the frame and its levels: the climb inside the frame, then each level's frame half and time half, starting functions one at a time | `brain.rs`: `Brain`, the orchestrator, as today; its body becomes this frame |
| `machine.rs`: `cover_pass`, `cover_in_election`, `cover_count`, `covered_offs`, `lift_relations`, `standing_over`, `layout_side`, `decide_kind`, `blocked`, `mark_refused`, `joined_chain`, `remove_forms`, `latest` | covering, lifting, the kinds decided, deadlocks refused, laps followed | `cover.rs`, an `impl Brain` block; the row writes it makes become batch calls (section 2) |
| `activation.rs`: `Act`, `ActRef`, `BoardRel`, `FrameRec`, `Inference`, `offset_between` | the open activations on the board | `board.rs`; the board is held by the thalamus, and it replaces `memory.rs` |
| `neighborhood.rs`: `neighborhood`, `stands`, `is_open`, `under`, `covered_ends` | what an activation sees, and what still stands | `board.rs` |
| `function.rs`: `Rel`, `Func`, `Layout`, `Dictionary`, poses and phases | the dictionary: every function, its relations, nodes and interneuron, and the poses | `dictionary.rs`; the dictionary is held by the thalamus |
| `recognize.rs`: `find`, `fit`, `fits_of`, `group`, `answer`, `worth_in`, `end_counts`, `calls_pair`, `alone_over`, `leave`, `retire` | recognition: finding and fitting functions, and grouping, which counts over rows | `recognize.rs`, an `impl Brain` block; its counts are answered by the neurons in batch calls (section 2, B9) |
| `election.rs`: `elect`, `accept`, `reuse`, `rank`, `price` | the election, and allocation or reuse of an interneuron | `election.rs`, an `impl Brain` block; allocation becomes a batch (B7) |
| `connect.rs`: `connect_all`, `add_earnings`, `walk`, `infer_all` | the connect call, rewards, exploration, the infer call | `connect.rs`, an `impl Brain` block; both calls become batches (B4 to B6) |
| `expansion.rs`: `expand_from`, `expands_to`, `func_for` | expansion of a call to base symbols | `expansion.rs`, an `impl Brain` block |
| `selection.rs`: `select`, `inferred_ahead`, `match_all`, `stood` | selection of the frame's output | `selection.rs`, an `impl Brain` block |
| `death.rs`: `reap`, `decommission`, `latest_standing_over` | the ledger and decommissioning | `death.rs`, an `impl Brain` block; decommissioning becomes one batch per reap (B8) |
| `neuron.rs`: `Neuron`, `Row`, `Nb`, `RowInput`, `ActReturn`, the relation index, the frame's relation index, connections | one neuron: its history of rows, its relation index, its connections | `neuron.rs`, replacing the neuron there now |
| `connection.rs`: `Conn` | a connection: strength and earnings | `neuron.rs`, beside the neuron |
| `thalamus.rs`: `Thalamus`, `route`, `push`, `process_level`, `forget`, and `get`, `get_mut`, `Index`, `iter` | the dispatch, and the reach to one neuron by id | `thalamus.rs`: the dispatch, the dictionary, the board, and a registry of each neuron's kind, dimension, bucket, channel, axes and whether it is alive; the reach to one neuron by id does not come over |
| `region.rs`, `column.rs` | fan-out over columns | `region.rs`, `column.rs` as they are, with the batch calls of section 2 in place of today's |
| `types.rs` | coordinates, offsets, poses, phases, kinds, policies, rewards, inputs | `types.rs`, under the names `types.rs` uses where both have one (`DimensionId`, `FrameNumber`, `ChannelId`, `NeuronId`) |
| `examples/` (`xy`, `stocks`, `channels`, `tap`, `text`, `shapes`, `mnist`), `benchmarks/` (`run.sh`, `README.md`, `output/`) | the benchmarks | `brain/brain-core/examples` and `brain/brain-core/benchmarks`, the same paths; each example reads the `Brain` only through its public calls |

**The thalamus holds the dictionary and the board.** It already holds the central maps about neurons, and these
are the machine's maps about functions and open activations. Neither is per-neuron state, so neither crosses into
the columns: they are fields of the thalamus beside the regions, and the `Brain`'s logic reads and changes them
there directly. Grouping and the election change the dictionary many times in one level, and every fit reads it,
so they must stay plain fields and not sit behind a batch call. If the borrows of a field beside the regions get in
the way, they move to the `Brain`, and the plan says so where it happens.

**Routing.** The thalamus routes as it does today: region `id mod R`, column `id mod C` within the region. The
prototype routed differently. Placement changes no result, because every batch call hands its results back in the
order its tasks came, as the prototype's `process_level` does. Results gathered in column order would change
results: `activate_level` keeps the stronger of two returns of one edge and, on a tie, the one that came first.

**Options.** The constructor takes every option: the depth of a history `H`, the neighborhood policy, the action
policy, learning on or off, regions and columns. Regions and columns default to one region of one column. The
examples read the options from their command lines and pass them in. The core reads no environment variable and
does not ask for the number of cores; the prototype's `Machine::new` did.

**Dead code is left out.** No path that no benchmark runs comes over: the relation-standing switch off, the
`round` reading of a closing candidate, the crossing charge, a level run without electing the dictionary's fits
first, the counters kept only to be printed, the trace lines, `watch_lines`, `note_change`, and the neuron's
methods nothing calls (section 2).

**Code style.** The style is the one in `brain/brain-core` today. Every function name starts with a verb: the
prototype's noun names (`rows_of`, `strength`, `end_counts`, `answer`, `latest`, `under`, `func`) are renamed. Doc
comments are `///`, a one-line header with the mechanism distributed into the body next to the code it describes.
Long lines are broken up. Comments cite the spec rule they implement and never a switch number.

## 2. Every single-neuron access, and the batch call that replaces it

The thalamus has no call that returns or changes one neuron. The prototype's `Thalamus::get`, `get_mut`, `Index`,
`IndexMut` and `iter` do not come over. Every access below goes through a batch call that takes a list, routes it,
runs the columns in parallel, and hands back results in the order the list came.

### A. Fixed facts: read from the registry, no crossing

These read a neuron's kind, dimension, bucket, channel, axes or whether it is alive. Those facts are set when the
neuron is created and change only at decommissioning, which the machine itself decides. The thalamus keeps them in
its registry, as it keeps base neuron facts today, and the `Brain` reads them there without reaching a column.

| file | function | reads |
|---|---|---|
| `machine.rs` | `process_frame` (weak activations, `bare_after_l0`), `new_act`, `frame_up_levels` | dimension, kind, axes, bucket |
| `machine.rs` | `frozen_half`, `funcs_never_called`, `height_counts`, `describe_neuron` | alive, kind, dimension, bucket |
| `connect.rs` | `share_now`, `walk` | kind, channel, dimension |
| `selection.rs` | `inferred_ahead`, `select`, `match_all` | alive, axes, dimension, bucket, kind |
| `expansion.rs` | `expand_from` | alive, kind |
| `election.rs` | `reuse`, `accept` | alive |
| `death.rs` | `reap`, `decommission` | alive; `decommission` sets it, in the registry |
| examples | `m.thalamus[..].bucket`, `.dim`, `.alive`, `m.thalamus.len()` | the same, through `Brain` calls |

### B. Neuron state: batch calls

**B1. Activating a level.** `machine.rs` `activate_level` already hands every neuron its rows in one
`process_level` call. It stays one batch call. Hands over: per neuron, each activation's row input: its record,
its pose, its neighborhood. Comes back: per activation, each neighbor with the strength of its relation, in task
order. One thing joins it. After the call, the prototype makes each row written now take the relations lifted to a
call before the row existed (`lift_pending`, through `row_takes`), and the strengths already returned do not count
them. In the port these pending lifts travel in the same task, as neighbors the neuron adds to the row after it has
computed the strengths it returns. One call, the same order of effects.

**B2. Activating a level with learning off.** `activate_level`, frozen branch: `thalamus[n].peek(rows)`, one
neuron after another. It becomes B1's call on a `Brain` built with learning off: each neuron computes the strengths
`activate` would return and writes nothing.

**B3. Lifting a relation to a row.** `machine.rs` `row_of` and `row_takes`, called from `lift_relations` (inside
`cover_pass` and `cover_in_election`) and from `activate_level`. `row_of` reads one neuron's history to find the
row of an activation, or for a call that took a lap, the row of the record it grew from. `row_takes` checks
whether that row already holds the call at that offset, and if not adds it through `add_followed`. It becomes one
batch call, `write_lifts`. The machine collects the lifts as `lift_relations` produces them and hands them over
once at the end of each election and once at the end of each cover pass. Hands over, per lift: the neuron; the
records that stand for the row (the activation, and the records of its frame that a lap resolves to the same
call); the call's record and the records that resolve to it; the call's pose; the offset. The machine works out
the records at the moment of the lift, from its laps (`grown`), so a lap taken later in the election changes
nothing. The neuron finds the row, adds the call where the row does not hold it, and updates its relation index.
Comes back: per lift, whether a row was found. For each lift with no row, the machine queues the pending lift
under the prototype's condition (inside the frame, a call of this frame not yet activated).

The flush is at the end of each election because the election reads no rows and the grouping that follows it in
the same level does. The prototype wrote each lift as it came. Writing them when the election ends gives the
grouping the same rows.

**B4. What followed.** `connect.rs` `connect_all`: `thalamus[n].add_followed(me, nbs)`, per open activation past
age 0. It becomes the first part of one batch call, `connect`. Hands over, per row: the neuron, the row's record,
the neighbors that stood on this frame's apex within reach, each with its pose and offset. Comes back: nothing.

**B5. Connections and rewards.** `connect.rs` `connect_all`, `add_earnings`, `walk`:
`thalamus[holder].connections.entry(..)`, one write after another, and the walk to the next untried action
where an estimate turns negative. It becomes the second part of `connect`. Hands over, per holder and in the
prototype's order: each write (own pose without laps, target pose without laps, offset, this frame's share), each
share for an earlier frame, and for each target the ordered list of base actions the walk would try. The machine
computes that list from the dictionary through expansion, since the neuron has no dictionary. The neuron applies
its writes and shares in that order, and walks where its own estimate turns negative. Its order of writes is the
prototype's, so a connection the walk creates and a later write in the same frame meet as they did. Comes back:
nothing. Holders are independent of one another, so the columns run them in parallel.

**B6. Inferring.** `connect.rs` `infer_all`: reads `thalamus[a.neuron].connections` per activation of the frame.
It becomes one batch call, `infer`. Hands over: per activation at age 0, its neuron and its pose without laps.
Comes back: per activation, each connection of that pose (target, offset, strength, estimate), in the order the
neuron's map holds them. The neuron keeps the prototype's map type, key and order of insertion, so selection sums
the same numbers in the same order.

**B7. Creating neurons.** `machine.rs` `declare_dim` and `election.rs` `accept`: `thalamus.push(Neuron::..)`, one
neuron at a time. The id is allotted at once, by the thalamus, with the neuron's facts in the registry, so the
dictionary and the board use it in the same step as in the prototype. The neuron itself is created by one batch
call, `create_neurons`, handed over at the end of the channel declaration and at the end of each election, before
B3's lifts of that election and before any call that reaches the new neuron. Hands over: per neuron, its id, kind,
dimension, bucket, channel, axes and `H`. Comes back: nothing.

**B8. Decommissioning.** `death.rs` `decommission`: sets `alive` on one neuron, clears its history, relation
index and connections, then `thalamus.forget(dead)` on every neuron. Nothing in `reap` reads a neuron's rows or
connections, so the machine sets `alive` in the registry as it decides, collects the decommissioned neurons and
their poses, and hands them over in one batch call, `decommission`, at the end of `reap`. Hands over: the neurons
to clear, and the poses to forget. Each column clears its own decommissioned neurons and removes those poses from
every row, relation index and connection of its living neurons. Comes back: nothing. The removals of several
decommissionings commute, so one call gives what one call each gave.

**B9. Counting during grouping.** `recognize.rs`: `rows_of` (read by `held_by`, `returned_for`, `rows_count`,
`answer`, `rel_through`, `alone_over`), `end_counts_read` (reads `strength` and `now_strength` off the relation
index), and `calls_pair` (reads an interneuron's whole history). Every one reads one neuron at a time, many times
per grouping. In the port the neurons do the counting. Each of these reads becomes a query the neuron answers over
its own rows, in its column: the end counts of a relation at one pose; `with` and `without` of a relation at a
node given the function's other relations there; whether a pose's rows hold a relation, and how many rows it has;
how many rows hold a relation bare or through a call; and, for a candidate that would close a cycle, the
interneuron's rows with it, without it and not yet decided. Every query carries the frame number, because whether a
row has decided a forward offset depends on it, and the poses the neuron cannot know, such as the poses of a call's
interneuron. The part of `calls_pair` that reads this frame's board stays on the machine.

The prototype states that no row changes while a grouping runs, so an answer is the same whenever it is asked.
That makes the timing of the queries free, and only the batching is to be designed. The grouping is a run of
decisions: each candidate weighed against the functions it could join, each relation tested for leaving, each
relation tested for starting a function. The queries of one decision are known when the grouping reaches it, and
not before, because the decisions before it change the functions. So the port gathers the queries of each decision
into one batch call, `count_relations`: hands over the queries, comes back one answer per query, in order. The
grouping keeps the prototype's memos as a cache of answers for the length of the grouping, so a query already
answered never goes out again. The results cannot differ from the prototype's as long as each query counts what
the prototype's read counted.

Whether this is faster is measured, not assumed. The counting runs in parallel and no row is copied, but a
grouping makes one round trip to the columns per decision instead of none. Stage 1 runs it against the plain
reading, the rows of the grouping's neurons read once in one batch and counted on the machine, on the stage's
benchmarks, and again at MNIST. If the counts in the neurons change any result, the work stops and the cause is
reported. If they are slower, the work stops and the two run times are reported.

**B10. What the benchmarks print about neurons.** `machine.rs`: `stats.connections` (the sum of every living
neuron's connections), `twice_counts` (reads each interneuron's history), `describe_neuron` (reads one neuron's
rows and connections); and the examples, which read `m.thalamus[..].history` and `.connections` directly. Where
an example prints these as part of its results, they become batch calls it reaches through the `Brain`:
`count_connections` (comes back: the total), and `inspect_neurons` (hands over the neurons; comes back each one's
row count and its connections, sorted as `describe_neuron` sorts them). Where they print only a diagnostic, they
are left out with it.

**Not ported: methods nothing calls.** `neuron.rs` `replace_seen`, `mark_saved`, `add_extra`, `discount`,
`add_beside` and `row_nbs` are never called. The row fields only they wrote, `saved`, `extra` and `discount`, are
always empty, so a row's relations are its neighbors' relations alone. None of them comes over.

## 3. What happens to what is in `brain/brain-core` and beside it now

The code there now is the model before functions and calls: spatial and temporal pattern neurons, contexts,
minting on errors, voting.

| part | what happens | why |
|---|---|---|
| `brain.rs` `Brain` | **kept as the orchestrator, body replaced.** The constructor with its options, `process_frame`, `save`, `load`, `save_context` and `load_context` stay; the spatial and temporal phases, the voting, `learn`, the accuracy tracking and the consensus modes go | the frame is the prototype's |
| `thalamus.rs` | **kept as the dispatch, contents replaced.** Stays: owning the regions, routing, id allotment, channel and dimension declaration. Comes in: the dictionary, the board, the registry. Goes: the coordinate-keyed base neuron map, parents and levels, channel neighbor sets and positions, `spatial_capacity`, the death ledger, the spatial and temporal level calls, corrections, commits, action wiring, vote collection | the batch calls of section 2 replace them; the ledger moves to the machine with `death.rs` |
| `region.rs`, `column.rs` | **kept, batch calls replaced** | the fan-out is the structure; what it fans out changes |
| `neuron.rs` | **replaced** by the prototype's neuron | rows, a relation index and connections replace patterns and contexts |
| `context/` | **removed** | there are no contexts |
| `memory.rs` | **replaced** by the board (`board.rs`) | the board holds the open activations |
| `diagnostics.rs` | **replaced** by the frame's counts (`FrameStats`) | the benchmarks print those |
| `types.rs` | **replaced** by the prototype's types, under the names this file uses | |
| `backup.rs` | **replaced, as part of the port** (stage 5) | every part of the saved state changes shape; the new state is saved and loaded, and the frozen benchmarks run on a loaded copy |
| `quantizer.rs` | **rewritten after the port**, with the Node addon (stage 12), where the jobs need it | the core takes buckets, as the prototype does |
| `brain-napi` (`src/lib.rs`, `index.js`, `index.d.ts`), `brain/src/index.js` | **rewritten after the port** (stage 12) | the tests run in JS, calling the Rust core through this addon; it binds the old `Brain`, so it leaves the workspace's build during the port and comes back bound to the ported `Brain` |
| `apps/` jobs (`mnist`, `stocks`, `text`) | **rewritten after the port** (stage 13): one job per benchmark, through the Node addon | the tests run in JS; their data (`apps/stocks/data/1D`, `apps/mnist/data`) stays, since the benchmarks read it |
| `apps/db` | **left out** | it runs no benchmark |

**Backups.** A backup is a folder of CSV files, as today: channels, dimensions, the registry, each neuron's rows,
its connections, the dictionary (each function's nodes, relations, origin and interneuron, and the pose table),
the ledger and the functions kept readable after they left. `save_context` and `load_context` save and load the
board. The relation index is not saved: it is recounted from the rows on load, since it is a count of them.

**Freezing.** The prototype freezes the running machine, and no example unfreezes it. The port freezes the way
the `Brain` does now: the example saves the brain and its context, and loads them into a new `Brain` built with
learning off. That `Brain` runs the frame as trained and changes nothing of itself, as the prototype's frozen
machine does: no row is written, no function starts, joins, moves or leaves, no interneuron is allocated, no
connection or reward is written, nothing is decommissioned. The poses it names while it runs are kept apart, as the
prototype keeps them. `two-and-three` then proves the backup as well as the frozen frame.

**Anything not in the benchmarks is left out.** Lines and squares are in the benchmarks: `hline`, `vline`,
`filled-square` and `hollow-square`, all run by the example `shapes`. The prototype's examples `line`, `square`,
`simple`, `slope` and `split`, its tests in `tests/machine.rs` and its `testset` are not, and do not come over.

## 4. The order of work

Every benchmark runs the whole frame, so the logic comes over whole in stage 1. The stages that follow each add
the code paths and batch calls that only their benchmarks reach, and each is proven by those benchmarks before the
next starts. Each stage runs on one region of one column. A stage that does not match stops the work (section 5).

| stage | work | proven by |
|---|---|---|
| 1. The structure and the frame | `types.rs`, the registry, the neuron, the batch calls B1, B4 to B7 and B9, the dictionary, the board, the frame with its levels, recognition, grouping, the election, covering and lifting in time, connect, infer, selection, `inspect_neurons`; the example `xy` | `alternation`: 30/30, one function, last change at frame 8 |
| 2. Text | the examples `tap` and `text` | `tap` 37/37, `text` 31/35, `text-without-space` 20/35 |
| 3. Stocks on laps | decommissioning (B8), rewards over spans and the walk, `count_connections`; the examples `stocks` and `channels` | `six-bar-lap` 165/178 (11 functions, frame 96), `kgc-gld`, `two-six-bar-laps`, `six-and-five-bar-laps`, `three-stocks`, `four-stocks` |
| 4. Space | covering inside the frame, lifting to rows (B3) and pending lifts; the example `shapes` | `hline`, `vline`, `filled-square`, `hollow-square` |
| 5. Backups and the frozen frame | `backup.rs`, `save_context`, `load_context`, learning off (B2) | `two-and-three` |
| 6. MNIST | the example `mnist` | `mnist-1x60`, then `mnist-10x20`, `mnist-100`, `mnist-255`, `mnist-10x300`, `mnist-1000` |
| 7. Long runs | none expected | `three-stocks-5-laps`; `ten-stocks` with its profit |
| 8. Parallel | none expected | every benchmark again on more columns, with the same results |
| 9. Your verification | none | you verify the port; the rules wait for it |
| 10. Rule one | section 6 | every benchmark, against the figures of `a30fb88` |
| 11. Rule two | section 6 | every benchmark, against the figures of `a30fb88` |
| 12. The Node addon | `brain-napi` bound to the ported `Brain`: its constructor options, channel declaration, `process_frame`, the reads the jobs print, `save`, `load`, `save_context`, `load_context`; `quantizer.rs` where a job needs it; built with `brain/build.ps1` | `alternation` from a JS job |
| 13. The jobs | one JS job per benchmark in `apps/`, printing the same result lines as its Rust example | every benchmark from its JS job, with the same results as from the Rust example |

Each benchmark's expected results are in `benchmarks/README.md` and its full output in `benchmarks/output`.

## 5. How results are checked

**The check is the results.** The traces and the diagnostic counters do not come over, so the outputs are not
compared byte for byte. What is compared is each benchmark's result lines: the right counts, per channel and per
lap, cycle, showing or episode where the example prints them; the neurons and the functions; when the dictionary
last changed; the dictionary itself where the example prints it; and the profit. The outputs print the dictionary
with neuron and function ids, so matching it means the same functions, built in the same order. `run.sh` keeps
its form, with a filter per example that keeps only those lines, applied to the saved output and the new one alike.
The saved outputs are never edited.

**Who runs it.** I run each stage's benchmarks with `run.sh --only <name>`, one at a time, each to a file read when
it finishes. I report the results of the stage and stop until you say to go on.

**When one does not match.**

1. Find the first frame where the result lines part. Where the lines are too coarse to say, a dump of the
   dictionary after every frame, made by the example from the `Brain`'s public reads, is added to both builds for
   the run and the two dumps compared. The prototype builds from the scratch export, where the same dump is added.
2. Name the cause from that frame, and only from what the dumps show.
3. The cause is a port fault until shown otherwise. The usual suspects: results handed back out of task order; a
   hash map iterated in an order the prototype did not have; a sum taken in another order; a batch flushed later
   than the prototype's write was read; a count query that does not count what the prototype's read counted.
4. Fix the port, never the expected output. A saved output is never re-recorded to make a stage pass.
5. Where the batch structure itself cannot give the prototype's order, the work stops, and that one problem is
   reported with the frame and the dump that show it.

## 6. The two rules

Both are in the spec at `a30fb88`, and go on only after you have verified the port. Its commit message has the
figures they gave together on a test copy of the prototype at `61d9b31`; among them: one MNIST image shown alone
last changes at episode 15 against 25, ten MNIST images over 20 episodes right 191 of 200 against 187 with 275
functions against 220, three stocks of 12 bars 104, 108 and 107 against 101, 108 and 97. Each rule is its own
stage, measured on every benchmark against those figures. Some benchmarks have no figure there: ten stocks, MNIST
255 and MNIST 1,000 were not run on the test copy, and MNIST 100, MNIST over 300 episodes and `two-and-three` are
not named. Their figures under the rules are new. Once you accept a rule's figures, its outputs are recorded as the
benchmarks' saved outputs (`run.sh --record`).

**Rule one: a candidate that brings a new node joins only at an end, a node that holds one relation of the
function.** It goes in `recognize.rs`, where a candidate is made for a function: `candidate`, in the branch for a
relation whose far end is not a node of the function. There the node the candidate attaches at must hold exactly
one of the function's relations. A function of one relation has two such nodes, and a cycle takes no candidate at
all, as now. This rule reads only the dictionary, so it needs no batch call. It is tested both ways, and the two
results reported: the end checked once, when a function's candidates are collected; and the end checked again at
the moment of the join, against the function as it stands after the joins before it.

**Rule two: a relation lifted inside the frame is lifted in every earlier row too.** Every earlier row of the same
pose that holds the covered activation's pose at the same offset takes the call, at the call's offset. It goes in
`cover.rs`, in `lift_relations`, at the point where the row at the relation's other end takes the call (B3, inside
the frame only). There the machine records one more entry for a batch call, `lift_earlier_rows`. Hands over, per
lift: the neuron of the pose at the relation's other end; that pose; the covered activation's pose and its offset
from the other end; the call's record, pose and offset from the other end; and the record of the other end's own
row, so that only rows written before it are touched. Each neuron finds its rows of that pose, written earlier,
that hold the covered pose at that offset and do not yet hold the call, adds the call to each, and updates its
relation index. One neighbor per call, however many relations were lifted to it. Comes back: nothing.

The lifts go to the neurons in one batch at the end of each level first. The test copy wrote them one neuron at a
time, as they came, so the grouping later in the same level saw them. So the rule is then run again with the lifts
handed over with each election's lifts (B3), and the two results reported.

## 7. Open questions

None.
