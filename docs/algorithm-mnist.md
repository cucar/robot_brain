# Digits from pixels

A worked case for [algorithm.md](algorithm.md): an image channel, a digit action, and a machine that learns to
name what it sees. Nothing here is normative. It follows the definitions through what forms over one image,
what the levels leave on the apex, and how the digit is taught and then inferred.

---

# 1. The environment

**In.** Two channels. The first, `image`, is laid out over two spatial activation dimensions, `x` and `y`, and
time, each with radius `1` (D1, R4): a pixel sees the pixels beside it, and nothing further needs to be exact.
It declares one event dimension, `pixel`, with one bucket, ink, so the channel has one base neuron (D52). A
point with no ink reports nothing: there is no rest value (§3.3), so a blank region costs nothing and holds no
activation. Each image is one frame, and consecutive images are unrelated.

With two spatial dimensions a bare record is four numbers, the neuron, the frame, `x` and `y` (D12, D13). A
call's record is four numbers, plus one per relation that did not come true, one per value, and one per
position of a landed activation in a dimension no relation fixes.

The resolution is the environment's to raise. With 256 buckets the dimension reports a gray level at each point.
With three channels sharing the layout, each with one event dimension, it reports red, green and blue; neurons of
the three channels are neighbors at offset zero in `x`, `y` and time (D5), so a function names across them like
anything else.

**Out.** The second channel, `digit`, has time as its only activation dimension and declares one action dimension,
`digit`, with ten base actions, `digit-0` to `digit-9`: a digit is said once per frame, at no place in the image,
so it has no `x` and no `y` (D1). The action runs in the frame after the image (R29), and the reward that arrives
with that frame is positive when the action named the image's digit and negative when it did not. It is scoped
to the digit channel and to that one frame (D35).

**Policy.** The machine declares the action policy as the least bad (D51): an image is named whatever the
estimates say, and where the machine is silent the world names it. It declares the neighborhood policy the
stocks case declares (D50): no action is anyone's neighbor, and only an event or an interneuron connects, to an
action alone. Consecutive images are unrelated, so an apex connecting to the next image's apex would learn
nothing, and a digit named inside a function would scatter its estimate over whatever covered it. Under this
policy every connection is to a digit action, and a reward scoped to the digit channel reaches exactly those
(R33).

# 2. What forms over an image

| What stands | its reach | what it sees, and what forms |
|---|---|---|
| a pixel | 1 in `x`, in `y` and in time (D4) | An inked pixel sees the inked pixels within one of it in `x` and in `y`, in this frame and the one before. Each neighbor gives one relation per dimension (D34, R21): ink at `x:−1`, `0` or `+1`; at `y:−1`, `0` or `+1`; at `t:0` or `−1`. Ink's history holds each in many rows. Grouping merges the relations whose rows intersect into functions of fixed nodes, all of them ink (R50), and keeps each whose margin is positive (D30). |
| an interneuron over pixels | 1 beyond the farthest pixel it stands for, from its centroid (D4) | It sees the interneurons within reach, the pixels under them (D47) and the pixels nothing covered. Its history, over its activations across images, holds relations to them, one per dimension. Grouping, finding and the election run as at the base, and a relation that lands on an interneuron covers its record (D22). |
| an interneuron above | the same, one more each level | Levels run until one fires nothing (R26). |

**What a function saves.** Take a function of `k` relations from ink's history, each landing on a pixel pinned in
that one dimension, the pixel's other two positions written in the call (D15, D22):

| | numbers |
|---|---|
| A, the pixel's record and each landed pixel's record | `4(k + 1)` |
| B, the call's record, plus two per landed pixel for the positions no relation fixes | `4 + 2k` |
| savings per row | `2k` |
| entry in the dictionary: `k` relations, ink named once | `k + 1` |
| margin at two rows | `3k − 1` |

Where two relations land on one pixel, it is pinned in both and one position is written, so it saves 3; pinned
in all three dimensions it saves 4. A relation that lands on nothing is written with the call, one number (D12).
The margin is positive at two rows for any `k`, and ink's history holds such rows by the hundred.

**What grouping leaves.** Merging two relations lowers the cost of the rows that hold both: one call covers the
pixel and two neighbors and saves 4, where a second function could not cover the pixel again (D28) and would
save nothing. In a row that holds one of the two, the merged function fits with one relation that did not come
true and saves 1. So the relations whose rows intersect merge, and what forms is a few functions from ink's
history, each a loose arrangement: ink to the left, ink to the right, ink above, ink below, each pinned in the
one dimension its relation names, the call's record carrying the rest. A function says a neighbor stands at
`x − 1`; where in `y` it stands is written each time. Not pairs, corners and strokes: those pin a neighbor in
two dimensions, and a relation pins in one.

**The election tiles the image.** Each kept function is placed at every ink activation it was found from (D31),
and the greedy cover takes the placement with the highest coverage over price each round, a placement's
coverage falling as earlier rounds cover its neighbors (D28, R24). What is left is a tiling: placements that do
not overlap, each covering a pixel and the neighbors its relations landed on, and the pixels no placement pays
for standing bare (R27). A placement at the edge of a stroke has relations that did not come true, one number
each, and is taken where it still pays.

**Time.** The relations at `t:−1` land on the previous image's pixels, nearly all of them under that image's
calls. A relation that lands on some of what a call covers is credited nothing (R23, D47), so those relations
lower no cost, grouping does not keep them, and the functions that form are over `x` and `y` within the frame.

**Above.** Each accepted function's interneuron fires at the centroid of the pixels it covers, rounded half
toward the smaller position (D53), and sees one beyond its farthest pixel (D4): the interneurons beside it. A
relation between two interneurons in one dimension, held in two rows of the first's history, is grouped and
priced the same way, and covering the second's activation saves its whole record (D22). The election tiles the
level with what pays, and the next level runs over those interneurons, until a level fires nothing.

**Variables.** At the base every node is ink, so two functions never differ at a node and R52 makes nothing
there. Above, two functions that differ in the interneuron at a node are generalized into one with a class
there, or a parameter where the same interneuron stands at two nodes within each (R52). The general function is
priced as a replacement of the two: each occurrence writes one number more, its value, and the dictionary writes
one entry fewer, so it joins only where the dictionary saving outweighs the occurrences, long functions with few
occurrences, and the specific functions stand otherwise.

**Translation.** A function is its relations, offsets between its nodes (D15), so the same arrangement at
another place in the image is another activation of the same interneuron (D11), and what is learned about it at
one place holds at every other.

# 3. The digit, taught and then inferred

**Taught.** In the first stretch the environment executes the digit action itself, in the frame after each
image: an action the world runs on its own is a lesson (R35). It arrives strong, with its reward. Every
uncovered activation of the image's frame, the interneurons at the top and the pixels nothing covered, has a
window of one frame (D9), is open at age 1, and connects to it (R31): a connection to `digit-7` at offset `+1`,
with the reward it earned, and to nothing else, since the policy admits nothing else (D50). An interneuron that
recurs over sevens accumulates exposures to `digit-7` and nothing else, so its estimate for it climbs; a pixel's
connections are a marginal over every digit it has stood in (R35).

**Inferred.** When the environment stops moving first, the voters are the apex of the image's frame (R46). Each
proposes at the frame ahead the digits its connections at `+1` name, each with its strength and estimate (R36).
In the digit dimension every digit some voter placed is a candidate. Each voter is one vote, split across the
digits it placed by strength; a candidate's estimate is the mean of the estimates its voters placed it with,
each weighted by that voter's share; and the candidate with the largest estimate is output (R47). An
interneuron whose connections name one digit puts its whole vote there; a bare pixel spreads its vote over
every digit it has stood in. Where every candidate is negative, the least bad runs (D51). The environment
executes the output, reports it with its reward, and the apex connects to what ran (R31). A wrong digit lowers
the estimates of the connections that chose it; where one turns negative, the walk wires the next digit in the
alphabet's order at strength 1 and earnings 0, and its estimate of zero outranks the negative one (R37). A newly
allocated interneuron has nothing for the frame ahead, so what it covers speaks in its place until something
has run under it (R48). Where no voter has anything, the world names the image, and the apex learns it (R35).

# 4. Where the levels stop

The levels end on an image when one fires nothing (R26): when the interneurons at the top hold no relation
twice over that pays. What stands there then is the image's description: interneurons for its parts, each at
the centroid of what it covers, and whatever pixels nothing covered. That is the apex that learns and votes
(R27), and the level at which it stands is nothing the machine was told.
