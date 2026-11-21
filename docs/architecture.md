# Architecture

How the machine of [algorithm.md](algorithm.md) is built: the layers from the Rust core to the jobs, the structure the
core runs on, where each part of the machine lives, and the step a frame goes through. The algorithm itself — what a
function, a call, a relation or the apex is, and the rules that govern them — is in the specification; this document
names the code that carries it out.

---

## The layers

| Layer | Where | What it does |
|---|---|---|
| The core | `brain/brain-core` (Rust) | The machine: neurons, the dictionary, the board, the frame step |
| The addon | `brain/brain-napi` (Rust, N-API) | Exposes the core to Node.js as the `Brain` class, with typed options, inputs, rewards and reads |
| The library | `libs/node` (`robot-brain`) | Re-exports the addon, with the `Job` class and the runner every job uses |
| The jobs | `apps/*/jobs` | Each declares its channels and runs its frames: stocks, text, MNIST, shapes, tap |
| The benchmarks | `apps/benchmarks` | Runs 23 jobs with fixed settings and compares them with their saved outputs in `docs/benchmarks/output` |

The addon is built with `brain/build.ps1` (or `build.sh`), which compiles `brain-napi` in release mode and copies the
library to `brain/brain-napi/brain-napi.node`.

---

## The core's structure

```
Brain
 └── Thalamus ── registry, dictionary, board
      └── Region × R ── Rayon, in parallel
           └── Column × C
                └── Neuron × n
```

- **Brain** (`brain.rs`) runs the frame step. It holds the configuration (the history depth, the neighborhood and
  action policies, whether it learns), the channels and dimensions, and the frame number.
- **Thalamus** (`thalamus.rs`) is the relay between the brain and its neurons. It holds what the machine as a whole
  keeps — the registry, the dictionary and the board — and every call that reaches a neuron.
- **Region** (`region.rs`) wraps its columns and fans their work out in parallel with Rayon.
- **Column** (`column.rs`) owns a partition of the neurons and runs the per-neuron work of a batch on them.
- **Neuron** (`neuron.rs`) holds one neuron's history, relation index and connections.

A neuron is routed by its id: to region `id % R`, then to column `id % C` within it. The layout is one region of one
column by default; `--regions` and `--columns` set it.

### Batch calls

No code outside a neuron reaches into it. Every access is a **batch**: the thalamus groups the tasks by neuron, fans
them out over the regions and columns, and returns the results in the order the tasks came. A neuron allotted since the
last batch is created in its column before the next batch runs.

| Call | What the neurons do |
|---|---|
| `activate_neurons` | write a row to their history for each activation, refresh the relation index, and return the relations they hold to their neighbors with their counts |
| `write_lifts` | take relations lifted to a call onto the call's row |
| `count_relations` | answer the counts grouping asks of them (how often a relation held, with and without another) |
| `connect_neurons` | record what followed each activation, with the reward earned, on their connections |
| `read_connections` | return their connections, for inference |
| `decommission_neurons` | let a dead interneuron go, and forget what named it |
| `inspect_neurons`, `export_neurons`, `import_neurons`, `count_connections` | reads for the jobs, and backup and restore |

---

## Where the machine lives

| Part | Where | What it holds |
|---|---|---|
| The registry | thalamus | each neuron's fixed facts: kind (event, action, interneuron), dimension, value, channel, the spatial axes it is laid out over, and whether it is alive |
| The dictionary | thalamus (`dictionary.rs`) | every function — its nodes and its relations — with its interneuron, the poses (a neuron in a phase), and each interneuron's functions |
| The board | thalamus (`board.rs`) | the frames still open, each with its activations: neuron, pose, position, reach, level, what it covers and what covers it, what fired it, and the relations its neuron returned at it |
| A neuron's history | neuron | its most recent activations, as rows, at most `h` of them |
| A neuron's relation index | neuron | every relation its rows hold, with how often it held — recounted from the rows when a backup is loaded |
| A neuron's connections | neuron | per pose and offset, what followed: a strength (times seen) and earnings (reward paid); its estimate is earnings over strength |

---

## The frame step

`Brain::process_frame` takes what the world reported — the events observed and the actions that ran, each a value of
a dimension at a position — and the rewards, and returns the machine's output for the frame ahead.

1. **Base activations.** Each reported value activates its base neuron, strong; a value only the last frame's
   inference named activates weak, where the report names nothing.
2. **The levels** (`climb_levels`), from the base up, level by level:
   - **activate** the level's rows (`activate_neurons`); the neurons return the relations they hold;
   - **find** the functions those relations may fit, and **fit** them (`recognize.rs`);
   - **count** what grouping needs (`count_relations`, read again until every count it asks is answered) and
     **group**: take a new function, or grow or generalize one, where naming saves more edges than it costs;
   - **elect** the calls to keep (`election.rs`), **allocate** interneurons for new functions, and **activate** the
     calls;
   - **cover** what the calls stand for, and **lift** to a call the relations of what it covers (`cover.rs`,
     `write_lifts`).
3. **Cover the board** once more after the last level.
4. **Connect** (`connect.rs`): each activation on the apex records what now stands on the apex after it, and the reward
   pays the connections written at the frames its span reaches.
5. **Infer**: each neuron on the apex reads its connections and infers what follows it, each with an estimate.
6. **Decommission** (`death.rs`) the interneurons whose last function has left the dictionary, once nothing can still
   read them. A brain that does not learn skips this.
7. **Select** (`selection.rs`): expand the inferred calls down to the base neurons they stand for (`expansion.rs`),
   match them against what stands, and resolve one winner per dimension — in an action dimension the action with the
   largest estimate, in an event dimension the event the most inferences name. The winners are the output.

---

## Settings

The brain's options are fixed when it is built:

| Option | Default | |
|---|---|---|
| `h` | 256 | the depth of every neuron's history |
| `policy` | `eventsToActions` | the neighborhood policy: which kinds of neuron may be each other's neighbors, and which may connect to which. `allOn` lets every kind do both; `eventsToActions` makes no action anyone's neighbor and lets only events and interneurons connect, and only to actions |
| `actionPolicy` | `leastBad` | what an action dimension outputs when every candidate's estimate is negative: the least bad, or `nothing` |
| `regions`, `columns` | 1, 1 | the layout |
| `learning` | true | a brain built with learning off runs each frame as trained and changes nothing of itself |

A frozen evaluation is a separate brain: the trained one saved, loaded into a brain built with learning off.

---

## Persistence

A **backup** is a folder of CSV files at `<jobDir>/backups/<label>/` (`backup.rs`):

| File | Holds |
|---|---|
| `brain.csv` | the history depth, the policies, the frame number, the counts, the reaches |
| `act_dims.csv`, `dims.csv`, `channels.csv` | the activation dimensions (time first), the neuron dimensions, the channels |
| `neurons.csv` | the registry, and per neuron the latest frame a row was written for |
| `rows.csv` | each neuron's history, oldest first |
| `connections.csv` | each neuron's connections |
| `functions.csv`, `retired.csv` | the dictionary, and the functions retired but still readable |
| `poses.csv`, `interneurons.csv` | the pose table, and each interneuron's functions |
| `ledger.csv`, `last_func.csv`, `reach_of.csv` | the decommissioning ledger, and per interneuron the function that fired it last and its reach |

The relation index is not written: it is recounted from the rows on load. A **context** — the board and what the
machine keeps about it — is a folder at `<jobDir>/contexts/<label>/`, loaded into a brain loaded from the backup saved
with it.

---

## The Node addon and the jobs

The addon's `Brain` class:

- **Building:** `new Brain({ h, policy, actionPolicy, regions, columns, learning })`, then `declareAxis(name)` and
  `declareChannel({ name, event, action, axes })` — a channel has an event dimension, an action dimension, or both,
  each with its number of values.
- **Running:** `processFrame(inputs, rewards)`, where an input is `{ dim, bucket, position }` and a reward
  `{ reward, channels, frames }`; it returns the winners for the frame ahead. `resetContext()` clears the board and
  keeps everything learned.
- **Reading:** stats, neurons, the dictionary (`getDictionary`, `readDictionaryFingerprint`, `readFunctionStamps`,
  `describeFunctions`), poses, the board (`getBoard`, `getLastFrame`, `countApex`), and neuron descriptions.
- **Saving:** `save`, `load`, `saveContext`, `loadContext`.

The `Job` class (`libs/node/src/job.js`) runs a job: it builds the brain from the job's options and the command line,
declares the job's channels or loads a backup that carries them, runs the job's frames, shows its results, and saves
what the command line asked for. `runJob(import.meta, MyJob)` at the bottom of a job file runs it when the file is the
entry point. The README lists the jobs and their options.
