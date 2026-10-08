# Digits from pixels

A worked case for [algorithm.md](algorithm.md): an image channel, a digit action, and a machine that learns to
name what it sees. Nothing here is normative. It follows the definitions through what forms over one image,
how the frontier closes the shapes the levels leave open, and how the digit is taught and then inferred.

---

# 1. The environment

**In.** Two channels. The first, `image`, is laid out over two spatial activation dimensions, `x` and `y`, and
time, each with radius `1` (D1): a pixel sees the pixels beside it, and nothing further needs to be exact. It
declares one event dimension, `pixel`, with two buckets, ink and no ink. A pixel with no ink reports nothing:
there is no rest value (D10), so a blank region costs nothing and holds no activation. Each image is one frame,
and consecutive images are unrelated.

The resolution is the environment's to raise. With 256 buckets the dimension reports a gray level at each point.
With three channels sharing the layout, each with one event dimension, it reports red, green and blue; neurons of
the three channels are neighbors at offset zero in `x`, `y` and time (D5), so a pattern names across them like
anything else.

**Out.** The second channel, `digit`, has time as its only activation dimension and declares one action dimension,
`digit`, with ten base actions, `digit-0` to `digit-9`: a digit is said once per frame, at no place in the image,
so it has no `x` and no `y` (D1). The action runs in the frame after the image (R29), and the reward that arrives
with that frame is positive when the action named the image's digit and negative when it did not.

**Policy.** The machine declares the neighborhood policy the stocks case declares (D1): no action is anyone's
neighbor, and only an event or a pattern connects, to an action alone. Consecutive images are unrelated, so an
apex connecting to the next image's apex would learn nothing, and a digit chunked into a pattern would scatter
its estimate over whatever covered it. Under this policy every connection is to a digit action, and a reward
scoped to the digit channel reaches exactly those (R33).

# 2. What forms over an image

| What stands | its reach | what it sees, and what forms |
|---|---|---|
| a pixel | 1 | An inked pixel sees the inked pixels at `±1` in `x` and `y`. The relations are the owner with one neighbor and two neighbors standing together (D47); the functions are pairs and corners, a child each. |
| a pair or a corner | 2 along what it spans, 1 across | A child sees the frontier within its reach: other children, and the pixels the base left uncovered. Strokes form, and a stroke of three pixels forms as a pair child beside the pixel it did not cover (D5). |
| a stroke, a part | twice what it spans (D4) | Strokes join into parts of digits, and parts into a digit's whole shape, each a level above the highest thing it covers (D2). |

Variables form where the shapes vary. An offset where different children have stood, the end of a stroke that
curves either way, is a class (D41); the same patch at two offsets, a symmetric digit, is a parameter (D44).
Time offsets recur in nothing, since the images are unrelated, so no pattern comes to name one: a part of the
previous image is a neighbor like any other, and it never pays.

# 3. The digit, taught and then inferred

**Taught.** In the first stretch the environment executes the digit action itself, in the frame after each
image (§3.5). It arrives strong, with its reward. Every uncovered activation of the image's frame, the children
at the top of the stack and the pixels nothing covered, is open at age 1 and connects to it (R31): a connection
to `digit-7` at offset `+1`, with the reward it earned. The shapes that recur over sevens accumulate exposures
to `digit-7` and nothing else, so their estimates for it climb; a pixel's connections are a marginal over every
digit it has stood in (R35).

**Inferred.** When the environment stops moving first, the frontier of the image's frame votes (R46). A child
standing for a shape seen only in sevens infers `digit-7` at a high estimate; a bare pixel infers the digit
that most often followed it, at a low one. The largest estimate is output, the environment executes it and
reports it with its reward, and the frontier connects to what ran. A wrong answer lowers the estimates that
chose it, and the walk (R37) tries the next digit where an estimate turns negative.

# 4. Where the levels stop

The stack ends on an image when no level above holds an activation (R26), which is when the shapes on the
frontier no longer recur in company worth a line. What stands there then is the image's description: a few
children for its parts, and whatever pixels nothing covered. That is the frontier that learns and votes, and the
level at which it stands is nothing the machine was told.
