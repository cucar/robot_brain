"""Split-MNIST vanilla-MLP baseline — the catastrophic-forgetting contrast figure.

A plain 2-layer MLP trained under the *identical* class-incremental protocol used by the
brain's Split-MNIST demo (docs/mnist-demos.md): 5 tasks x 2 classes ({0,1} {2,3} {4,5}
{6,7} {8,9}), strictly sequential, each task seen once and never revisited, no task IDs,
the output layer spanning all 10 classes throughout. After each task the net is frozen and
tested on the full 10-class test set to build a retention matrix. Standard SGD with no
continual-learning machinery collapses to ~20% average accuracy: each new task overwrites
the weights for the previous one. This is the baseline the brain's ~96% retention beats.

Dependency-light by design — NumPy only, reading the same gzipped IDX files the JS app uses
(apps/mnist/data). No torch, no training of the brain.
"""
import gzip, struct, pathlib, numpy as np

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
rng = np.random.default_rng(0)


def load(images_gz, labels_gz):
    with gzip.open(DATA / images_gz, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        x = np.frombuffer(f.read(), np.uint8).reshape(n, r * c).astype(np.float32) / 255.0
    with gzip.open(DATA / labels_gz, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), np.uint8)
    return x, y


Xtr, Ytr = load("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
Xte, Yte = load("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")

# 784 -> 256 -> 10, ReLU hidden, softmax output, manual backprop, plain SGD.
def init_weights():
    global W1, b1, W2, b2
    W1 = rng.normal(0, 0.05, (784, 256)).astype(np.float32); b1 = np.zeros(256, np.float32)
    W2 = rng.normal(0, 0.05, (256, 10)).astype(np.float32);  b2 = np.zeros(10, np.float32)


init_weights()


def forward(x):
    h = np.maximum(0, x @ W1 + b1)
    z = h @ W2 + b2
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return h, p


def accuracy(x, y):
    return float((forward(x)[1].argmax(1) == y).mean())


def train_task(classes, lr=0.1, epochs=3, batch=128):
    global W1, b1, W2, b2
    mask = np.isin(Ytr, classes)
    x, y = Xtr[mask], Ytr[mask]
    for _ in range(epochs):
        for i in rng.permutation(len(x)).reshape(-1, batch) if len(x) % batch == 0 else \
                [rng.permutation(len(x))[j:j + batch] for j in range(0, len(x), batch)]:
            xb, yb = x[i], y[i]
            h, p = forward(xb)
            p[np.arange(len(yb)), yb] -= 1; p /= len(yb)   # softmax-CE gradient
            gW2 = h.T @ p; gb2 = p.sum(0)
            gh = (p @ W2.T) * (h > 0)
            gW1 = xb.T @ gh; gb1 = gh.sum(0)
            W2 -= lr * gW2; b2 -= lr * gb2; W1 -= lr * gW1; b1 -= lr * gb1


# Upper bound: the SAME MLP trained jointly on all 10 classes (shuffled together).
# This is the number the sequential run collapses *from* — the ceiling, not the floor.
init_weights()
train_task([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
joint = accuracy(Xte, Yte)
print(f"Joint training (all 10 classes at once, same MLP): {joint * 100:.2f}%   <- the ceiling\n")

# Class-incremental: same MLP, same data, fed 2 classes at a time, strictly sequential.
init_weights()
retention = []
for t, classes in enumerate(TASKS):
    train_task(classes)
    retention.append([accuracy(Xte[np.isin(Yte, c)], Yte[np.isin(Yte, c)]) for c in TASKS])

print("Retention matrix  " + "   ".join(f"T{i}{str(c).replace(' ', '')}" for i, c in enumerate(TASKS)))
for t, row in enumerate(retention):
    print(f"  after T{t}        " + "   ".join(f"{r * 100:5.1f}%" for r in row))
seq = accuracy(Xte, Yte)
headline = float(np.mean(retention[-1]))
print(f"\n  Average accuracy after Task 4 (headline): {headline * 100:.2f}%")
print(f"  Overall test accuracy: {seq * 100:.2f}%")
print(f"\nThe collapse: {joint * 100:.1f}% (joint) -> {seq * 100:.1f}% (sequential), "
      f"a {(joint - seq) * 100:.0f}pp drop to the ~20% chance floor.")
