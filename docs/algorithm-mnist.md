# Digits from pixels

A worked case for [algorithm.md](algorithm.md): an image channel, a digit action, and a machine that learns to
name what it sees. Nothing here is normative. It follows the definitions through what forms over an image,
runs of one color first and the shapes over them after, through one small image level by level, and through
how the digit is taught and then inferred. The recognition here is strict: every length and every offset is
exact, and a shape at another size is another shape. The variables over laps and offsets, which will let one function
stand for a shape at every size, are not in this case.

---

# 1. The environment

**In.** Two channels. The first, `image`, is laid out over two spatial activation dimensions, `x` and `y`, and
time (D1): a pixel sees the eight pixels around it, and the nine at those positions and its own in the frame
before (D5). It declares one event dimension, `pixel`, with two buckets, black and white, so the channel has
two base neurons, and every point of the image fires one of them in every frame (D52). Nothing is silent: a
blank region is pixels of the background color, each an activation, and the relation of a stroke is where one
color stands beside the other. A diagonal is two pixels one position apart in `x` and in `y` at once, one relation
with a component in each (D34), and the arrangement of colors around a pixel is its neighborhood (D7). Each
image is one frame, and consecutive images are unrelated.

A bare pixel costs its relations to the pixels around it, one relation each, less those the others imply
(D12, D13, D55). A call costs nothing of its own, one per relation that did not come true and one per argument.

The resolution is the environment's to raise. With 256 buckets the dimension reports a gray level at each point.
With three channels sharing the layout, each with one event dimension, it reports red, green and blue; neurons of
the three channels are neighbors at offset zero in `x`, `y` and time (D5), so a function names across them like
anything else.

**Out.** The second channel, `digit`, has time as its only activation dimension and declares one action dimension,
`digit`, with ten base actions, `digit-0` to `digit-9`: a digit is said once per frame, at no position in the image,
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

**The election is per relation, and the longest run wins.** At a white pixel the four white line functions
fit, each reading the run through that pixel along its own direction, as many laps as hold (D54). A fit
covers the run's relations, one fewer than its pixels, for nothing, so a run of two covers one and is taken
where nothing longer covers it, and the longest run through a pixel covers the most and is taken first (D28,
R24). A call covers relations, not pixels: a row covers a pixel's relations along `x` and a column its
relations along `y`, so a pixel may be in both, and where two lines cross both are whole (D10). The two calls
then hold one edge between them, and the second taken pays it: a run of two crossing a taken line covers one
for that edge, and is not taken, while a run of three is (D28). Two runs of the same length along one
direction through a pixel share its relations and tie, and the older line function takes it.

**What a line is on the board.** A line's call is one activation of the line's interneuron, at the run's first
pixel, the least `x` of a row and the least `y` of a column (D37), in a pose that says its laps, one fewer than its
pixels (D53). The run's pixels are gone from the apex, and the call inherits every
relation they had that the fit did not cover: a pixel beside any pixel of the run now relates to the line, at
its offset to the line's position, so beside the middle and beside the end are two relations, as every relation
at the base is one offset (D10, D5). After the first level the board holds lines, each one activation with its
laps, and the pixels no run reached, bare: lone pixels.

**Strokes and sides, over the lines.** A line's interneuron sees what its pixels saw (D4), and its rows
hold relations to what stood there: another line at the offset between their positions, a lone pixel, the other
color's line beside it. Each at strength 3 is a function over the line call, taken where it covers its
relation (R24). Two parallel rows of one length one position apart are a row after a row at `(0, +1)`, a cycle
over the row interneuron in that pose, and its fit over two rows covers the relation for nothing. A stroke
three pixels thick is three rows and two laps of that cycle, one call. Two lines that share a pixel hold one
edge between them, at the offset between their positions (D10), and no function is built on that edge, since
the pixel already ties them (R50): a row and a column that meet at a corner, or cross, are two calls and never
one motif. What joins them is something that shares no pixel with either: a row beside a row, or the block
that stands between two columns. Each such pair at one offset is a motif of two calls, 1 for nothing, and a
lone pixel beside a line is a motif of the line and the pixel, the same. Which of these a line ends up under
is the election's again, by what each covers less its price. Motifs that share a call and hold together
merge, for the entry one function no longer pays (R54, D13),
and a function grows over the lines as far as its relations hold. The edge between two lines that share a
pixel costs nothing once a path of covered relations joins them (D55).

**Above.** Each level's calls are the next level's neurons: strokes beside strokes, lines held to the
blocks between them at the offsets the image puts there, until a level fires nothing (R26). A function at any level is its
relations, offsets between its nodes (D15), so the same arrangement at another position in the image is another
activation of the same interneuron (D11), and what is learned about it at one position holds at every other.

**Time.** At `t:−1` a pixel sees the previous image's calls and its bare pixels (D5). Consecutive images are
unrelated, so no relation across them recurs, nothing is built on them (R21), and the functions that form are
over `x` and `y` within the frame.

**Sizes are exact.** A relation to a line names the line in its pose, laps included (D34), so a line of four
and a line of five are two different ends, a motif over them two different functions, and a frame of another
size another frame. Nothing here rounds a length or an offset, and nothing stands for a shape across sizes:
that is what the variables over laps and offsets will add, a parameter where two sides must be the same length and a
class where a length may vary within a range. Until then the dictionary holds every shape at every size it
recurred at, and recognizes each exactly.

# 3. One image, level by level

A 5×5 image, black with a 3×3 white square in the middle: white at `x` and `y` from 1 to 3, black at the
sixteen pixels around it. Coordinates are `(x, y)`, with `(0, 0)` the top-left. Every function below has
recurred in earlier images, so each is in the dictionary with its interneuron.

**Bare.** Twenty-five pixels, joined by twenty-four relations, the rest implied (D55).

**Level 1: seven lines.** The black frame is four runs of five: the rows at `y = 0` and `y = 4`, the columns
at `x = 0` and `x = 4`. The rows are older and are taken first, four for nothing each. Each column shares a
corner pixel with each row, so its price is two edges, and it covers four: taken. The white square is three
rows of three and three columns of three. The rows are taken, two for nothing each; a column then shares a
pixel with each of the three rows, so its price is three edges against two covered, and it is not taken.
Nothing is bare.

| after level 1 | |
|---|---|
| on the apex | seven calls, each at its line's first pixel, in its pose |
| the cost | the four corner edges, and the edges among the calls that the pixels' relations became, one per pair of calls at one offset |

**Level 2: the white block.** The three white rows are two laps of a row after a row at `(0, +1)`, one call
over the three, 2 for nothing. Each row of the frame shares a corner pixel with each column, so nothing is
built between a row and a column (R50), and the four lines stand as they were.

| after level 2 | |
|---|---|
| on the apex | five calls: the four lines of the frame and the white block |
| the cost | the four corner edges, and the edges from the block to the lines |

**Level 3: the image.** The block stands at `(1, 1)`, the rows at `(0, 0)` and `(0, 4)`, and the columns at
`(0, 0)` and `(4, 0)`, the least position among what each landed on (D37). The block shares no pixel with any
line and relates to each at its offset. Each pair, the block and one line, was a function at its third
image, and the four pairs merged into one function of five calls and four relations, each move gaining the
entry a function no longer pays (R54, D13). It covers 4 for nothing: one call. The four corner edges are
implied by it, since each line is joined to the next through the block (D55).

| after the last level | |
|---|---|
| on the apex | one call |
| the cost | 0 |

A 6×6 frame makes lines of six, which are other ends for every relation, so its motifs and its image are
other functions, each learned over its own recurrences. The same square at another position in a larger image
is the same functions (D11).

**What the walk shows about the design.** A row and a column that meet are tied by the pixel they share, and
nothing is built on that tie, so the frame is not made of corners: it is four lines, each held at its offset
from the block they surround. Where two lines cross, both are whole, and the crossing is the one edge between
them. And every size is its own set of functions, which is the strictness the case starts from.

# 4. The digit, taught and then inferred

**Taught.** In the first stretch the environment executes the digit action itself, in the frame after each
image: an action the world runs on its own is a lesson (R35). It arrives strong, with its reward. Every
uncovered activation of the image's frame, the calls at the top and the pixels nothing covered, has a window
of one frame (D9), is open at age 1, and connects to it (R31): a connection to `digit-7` at offset `+1`, under
its own pose, with the reward it earned, and to nothing else, since the policy admits nothing else (D50). An
interneuron that recurs over sevens accumulates exposures to `digit-7` and nothing else, so its estimate for
it climbs; a pixel's connections are a marginal over every digit it has stood in (R35).

**Inferred.** When the environment stops moving first, the voters are the apex of the image's frame (R46). Each
proposes at the frame ahead the digits its connections at `+1` name, each with its strength and estimate (R36).
In the digit dimension every digit some voter named is a candidate. Each voter is one vote, split across the
digits it named by strength; a candidate's estimate is the mean of the estimates its voters named it with,
each weighted by that voter's share; and the candidate with the largest estimate is output (R47). An
interneuron whose connections name one digit puts its whole vote there; a bare pixel spreads its vote over
every digit it has stood in. Where every candidate is negative, the least bad runs (D51). The environment
executes the output, reports it with its reward, and the apex connects to what ran (R31). A wrong digit lowers
the estimates of the connections that chose it; where one turns negative, the walk wires the next digit in the
alphabet's order at strength 1 and earnings 0, and its estimate of zero outranks the negative one (R37). A newly
allocated interneuron has nothing for the frame ahead, so what it covers speaks for it until something
has run under it (R48). Where no voter has anything, the world names the image, and the apex learns it (R35).

**What votes for a digit.** The lines of a digit are its strokes, and a stroke two or three pixels thick is one
call, with its length in its pose. A connection is keyed by the pose, so a seven drawn larger makes lines of
other laps, other motifs and other voters than a seven drawn smaller, and each size is learned over its own
images. That is the strictness of this case: it recognizes what it has seen, exactly, and nothing it has not.
How the strokes stand to one another is what tells a seven from a one; where the hand was
unsteady, a stroke is several short lines rather than one, and what recurs across images at an exact offset
and length is less than the walk above suggests. That is the case's risk, and it is a measurement, not a
derivation. The variables over laps and offsets are what will let one line function vote for every length and one
motif for every size.

# 5. Where the levels stop

The levels end on an image when one fires nothing (R26): when the interneurons at the top hold no relation
twice over that pays. What stands there then is the image's description: calls for its parts, each in its
pose, and whatever pixels nothing covered. That is the apex that learns and votes (R27), and the level at
which it stands is nothing the machine was told.
