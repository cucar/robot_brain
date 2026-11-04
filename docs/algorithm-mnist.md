# Digits from pixels

A worked case for [algorithm.md](algorithm.md): an image channel, a digit action, and a machine that learns to
name what it sees. Nothing here is normative. It follows the definitions through what forms over an image,
runs of one color first and the shapes over them after, through one small image level by level, and through
how the digit is taught and then inferred.

---

# 1. The environment

**In.** Two channels. The first, `image`, is laid out over two spatial activation dimensions, `x` and `y`, and
time (D1): a pixel sees the eight pixels around it, and the nine at those places and its own in the frame
before (D5). It declares one event dimension, `pixel`, with two buckets, black and white, so the channel has
two base neurons, and every point of the image fires one of them in every frame (D52). Nothing is silent: a
blank region is pixels of the background color, each an activation, and the relation of a stroke is where one
color stands beside the other. A diagonal is two pixels one position apart in `x` and in `y` at once, one relation
with a component in each (D34), and the arrangement of colors around a pixel is its neighborhood (D7). Each
image is one frame, and consecutive images are unrelated.

A bare pixel costs its relations to the pixels around it, one relation each, less those the others imply
(D12, D13, D55). A call costs its own relation, the one between its two ends, one per relation that did not
come true and one per argument.

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

**Lines.** A pixel sees the eight around it, and each neighbor is one relation at its offset in `x` and `y`
(D34, R21). The relations that recur most are a color beside itself: white at `(+1, 0)`, at `(0, +1)`, at
`(+1, +1)` and at `(+1, −1)`, and the same for black, each at a strength in the hundreds in that color's
history. Each is a neuron after itself, so each is a cycle of one node (D15): a run of that color along that
direction, for as long as it holds. A second offset to advance by is another function, so there are four line
functions per color, eight in all, and nothing joins them: a relation to the other color beside a run holds
at a few of a run's pixels and fails at the rest, so its `without` outweighs its `with` and it never pays
(D57).

**The election is per pixel, and the longest run wins.** At a white pixel the four white line functions are
placed, each reading the run through that pixel along its own direction, as many laps as hold (D54). A fit
covers the run's relations, one fewer than its pixels, for the call's own relation, so a run of two covers one
for one and is not taken, a run of three two for one, and the longest run through a pixel has the highest
coverage over price and is taken first (D28, R24). A pixel is covered by one line: a fit that read through a
pixel a taken fit covers is read again without it, so a line crossing a longer one breaks there, into the
runs on either side, each taken if it still pays. Two lines may end at one pixel: an activation may be the end
of several calls, and a corner is exactly that (D8, D28). Two runs of the same length through a pixel tie, and
the older line function takes it.

**What a line is on the board.** A line's call is two activations, at the run's two ends, with the run's
length between them as its own relation (D53): a row's start at its least `x`, a column's start at its least
`y`. The pixels between are gone with their relations; the end activations keep the relations the end pixels
had to the outside, so a pixel beside a line's end sees the line there, and a pixel beside its middle sees
nothing (D10, D5). After the first level the board holds lines, each two ends and one relation, and the pixels
no run of three reached, bare: lone pixels and pairs.

**Strokes and corners, over the lines.** A line's interneuron sees one beyond what it covers at each end
(D4), and its activations across images hold relations to what stood there: another line's end at offset zero,
a parallel line one position over, a lone pixel. Each at strength 2 is a function over the line call, and a fit
that lands on a call covers its whole count (D22), so a function of two nodes over calls pays where a pair of
pixels does not: two parallel rows one position apart are a row after a row at `(0, +1)`, a cycle over the row
interneuron, and its fit over two rows covers the relation and both calls' counts, 3 for 1. A stroke three
pixels thick is three rows and two laps of that cycle, one call. Two lines meeting at a pixel are a motif of
two nodes at offset zero, the one line's end beside the other's, a corner, 3 for 1 over the two calls. A lone
pixel beside a line's end is a motif of the line and the pixel, 2 for 1. Which of these a pixel ends up under
is the election's again, by coverage over price, and a call's count is paid to one fit, so two corners cannot
both take one line.

**Above.** Each level's calls are the next level's neurons: strokes beside strokes, corners joined to corners
at the offsets the image puts between them, until a level fires nothing (R26). A function at any level is its
relations, offsets between its nodes (D15), so the same arrangement at another place in the image is another
activation of the same interneuron (D11), and what is learned about it at one place holds at every other.

**Time.** At `t:−1` a pixel sees the previous image's calls at their ends, and its bare pixels (D5).
Consecutive images are unrelated, so no relation across them recurs, nothing is built on them (R21), and the
functions that form are over `x` and `y` within the frame.

**Variables.** At the base every node is a color, so two functions never differ at a node and R52 makes
nothing there. Above, two functions that differ in the interneuron at a node are generalized into one with a
class there, or a parameter where the same interneuron stands at two nodes within each (R52). The general
function is priced as a replacement of the two: each occurrence costs one binding more, and the dictionary one
entry fewer, so it joins only where the entry saved outweighs the occurrences, and the specific functions
stand otherwise. Sizes and slants are offsets, and no variable holds an offset yet.

# 3. One image, level by level

A 5×5 image, black with a 3×3 white square in the middle: white at `x` and `y` from 1 to 3, black at the
sixteen pixels around it. Coordinates are `(x, y)`, with `(0, 0)` the top-left. Every function below has
recurred in earlier images, so each is in the dictionary with its interneuron.

**Bare.** Twenty-five pixels, joined by twenty-four relations, the rest implied (D55).

**Level 1: seven lines.** The black frame is four runs of five: the rows at `y = 0` and `y = 4`, the columns
at `x = 0` and `x = 4`. Each covers four for one. The corner pixels are the ends of a row and of a column at
once, and an activation may be the end of several calls, so all four are taken. The white square is three
rows of three and three columns of three, two for one each; the rows and the columns tie at every pixel, and
the older function, say the rows, takes them. Nothing is bare.

| after level 1 | |
|---|---|
| on the apex | seven calls, fourteen end activations |
| the count | seven own relations, and the relations among the ends that the others do not imply |

**Level 2: two corners and a stroke.** Each corner of the frame holds a row's end and a column's end at offset
zero, and the corner function, the row's end beside the column's start, fits at all four. A call's count is
paid once, so two of the four are taken, at opposite corners, each covering one row and one column, 3 for 1;
the other two are read again without the covered calls and cover nothing. The three white rows are two laps
of a row after a row at `(0, +1)`, one call over the three, 5 for 1.

| after level 2 | |
|---|---|
| on the apex | three calls: two corners, each over two sides of the frame, and the white block |
| the count | three own relations, and the relations among their six ends |

**Level 3 and 4: the frame, then the image.** The two corner calls stand at the frame's corners and relate at
the offset the frame's size puts between them; that is a function of two nodes over calls, 3 for 1, and the
frame is one call. The white block's ends stand one position inside the frame's, and the function of the frame
and the block, 3 for 1, is the image: one call, its own relation from `(0, 0)` to `(4, 4)`.

| after the last level | |
|---|---|
| on the apex | one call |
| the count | 1 |

A larger square, or a frame of another size, makes the same lines, corners and stroke, and differs only in the
offsets between the corner calls and in the own relations: another function at the third level, or another
own relation under the same one.

**What the walk shows about the design.** A call has two addressable ends, its start and its end. A corner's
call covers two lines and stands at two of their four ends, so the square is two corners and a relation
between them, not four corners: the two far ends of each corner's lines are gone with the call. And where
two lines cross, the pixel belongs to the longer, and the shorter breaks: a stroke through the middle of a bar
leaves the bar in two pieces and the stroke whole.

# 4. The digit, taught and then inferred

**Taught.** In the first stretch the environment executes the digit action itself, in the frame after each
image: an action the world runs on its own is a lesson (R35). It arrives strong, with its reward. Every
uncovered activation of the image's frame, the calls at the top and the pixels nothing covered, has a window
of one frame (D9), is open at age 1, and connects to it (R31): a connection to `digit-7` at offset `+1`, with
the reward it earned, and to nothing else, since the policy admits nothing else (D50). A call connects from
its start; its end, born at the age of its place, infers nothing (D9). An interneuron that recurs over sevens
accumulates exposures to `digit-7` and nothing else, so its estimate for it climbs; a pixel's connections are
a marginal over every digit it has stood in (R35).

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

**What votes for a digit.** The lines of a digit are its strokes, and a stroke two or three pixels thick is one
call whichever way it leans, with its length in its own relation and nothing above reading that. So a seven
drawn larger makes the same calls as a seven drawn smaller, and the same voters. Where the strokes meet, the
corners and the crossings are what tells a seven from a one; where the hand was unsteady, a stroke is several
short lines rather than one, and what recurs across images at an exact offset is less than the walk above
suggests. That is the case's risk, and it is a measurement, not a derivation.

# 5. Where the levels stop

The levels end on an image when one fires nothing (R26): when the interneurons at the top hold no relation
twice over that pays. What stands there then is the image's description: calls for its parts, each at its two
ends, and whatever pixels nothing covered. That is the apex that learns and votes (R27), and the level at
which it stands is nothing the machine was told.
