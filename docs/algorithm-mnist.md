# Digits from pixels

A worked case for [algorithm.md](algorithm.md): an image channel, a digit action, and a machine that learns to
name what it sees. Nothing here is normative. It follows the definitions through what forms over one image,
what the levels leave on the apex, and how the digit is taught and then inferred.

---

# 1. The environment

**In.** Two channels. The first, `image`, is laid out over two spatial activation dimensions, `x` and `y`, and
time (D1): a pixel sees the eight pixels around it, and the nine at those places and its own in the frame
before (D5). It declares one event dimension, `pixel`, with two buckets, black and white, so the
channel has two base neurons, and every point of the image fires one of them in every frame (D52). Nothing is
silent: a blank region is white pixels, each an activation, and the edge of a stroke is where black stands
beside white. A diagonal is two black pixels one step apart in `x` and in `y` at once, one relation with a
component in each (D34), and the arrangement of black and white around a pixel is its neighborhood (D7). Each
image is one frame, and consecutive images are unrelated.

With two spatial dimensions a bare record is four numbers, the neuron, the frame, `x` and `y` (D12, D13). A
call's record is four numbers, plus one per relation that did not come true and one per value.

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
| a pixel | 1 in `x`, in `y` and in time (D4) | An inked pixel sees the inked pixels among the eight around it and the nine at those places in the frame before. Each neighbor gives one relation, at its offset in `x`, `y` and time (D34, R21): ink at `(+1, 0)`, at `(+1, +1)`, at `(0, −1)`, and so on. Ink's history holds each at a strength in the hundreds. Grouping merges the relations that hold together in ink's history into functions of fixed nodes, all of them ink (R50), and keeps each whose worth is positive (D57). |
| an interneuron over pixels | 1 beyond the farthest pixel it stands for, from its origin (D4) | It sees the interneurons within reach and the pixels nothing covered, and nothing under a call (D5). Its history, over its activations across images, holds relations to them, one per neighbor. Grouping, finding and the election run as at the base, and a fit that lands on an interneuron as a node covers its whole count (D22, D54). |
| an interneuron above | the same, one more each level | Levels run until one fires nothing (R26). |

**What a function saves.** Take a function of `k` relations from ink's history, each landing on a pixel at
the offset it names in `x` and `y` (D15, D22):

| | numbers |
|---|---|
| A, the pixel's record and each landed pixel's record | `4(k + 1)` |
| B, the call's record | `4` |
| savings per occasion | `4k` |
| entry in the dictionary: `k` relations, ink named once | `k + 1` |
| worth at strength 2 | `7k − 1` |

A relation that lands on nothing is written with the call, one number (D12). The worth is positive at strength
2 for any `k`, and ink's history holds such occasions by the hundred.

**What grouping leaves.** Merging two relations lowers the cost of the occasions that hold both: one call
covers the pixel and two neighbors and saves 8, where a second function could not cover the pixel again (D28)
and would save nothing. On an occasion that holds one of the two, the merged function fits with one relation
that did not come true and saves 3. So the relations that hold together in ink's history merge, and what forms
is the arrangements of ink around a pixel that recur: a pixel with ink on both sides, a bend, the end of a
stroke, each neighbor at its own offset in `x` and `y`. Pairs, corners and pieces of stroke, since a relation
fixes its neighbor in every dimension at once.

**The election tiles the image.** Each kept function is placed at every ink activation it was found from (D31),
and the greedy cover takes the placement with the highest coverage over price each round, a placement's
coverage falling as earlier rounds cover its neighbors (D28, R24). What is left is a tiling: placements that do
not overlap, each covering a pixel and the neighbors its relations landed on, and the pixels no placement pays
for standing bare (R27). A placement at the edge of a stroke has relations that did not come true, one number
each, and is taken where it still pays.

**Time.** At `t:−1` a pixel sees the previous image's calls and the pixels nothing covered there, and nothing
under a call (D5). Consecutive images are unrelated, so no relation across them recurs, nothing is built on
them (R21), and the functions that form are over `x` and `y` within the frame.

**Above.** Each accepted function's interneuron fires at its origin, the covered pixel least in `x` and then
in `y` (D15, D53), and sees one beyond its farthest pixel (D4): the interneurons beside it and the pixels
nothing covered, and nothing under a call. A relation between two interneurons at an offset, at strength 2 in
the first's history, is grouped and priced the same way, and a fit landing on the second's activation covers
its whole count (D22). The election tiles the level with what pays, and the next level runs over those
interneurons, until a level fires nothing.

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
its origin, and whatever pixels nothing covered. That is the apex that learns and votes
(R27), and the level at which it stands is nothing the machine was told.
