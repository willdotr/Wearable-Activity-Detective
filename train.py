"""Train a small MLP (numpy) on UCI HAR, compare to a baseline, export for the website."""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "UCI HAR Dataset")
OUT = os.path.join(ROOT, "docs", "data")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

def load(split):
    X = np.loadtxt(os.path.join(D, split, f"X_{split}.txt"), dtype=np.float32)
    y = np.loadtxt(os.path.join(D, split, f"y_{split}.txt"), dtype=int) - 1
    s = np.loadtxt(os.path.join(D, split, f"subject_{split}.txt"), dtype=int)
    return X, y, s

def signals(split):
    axes = []
    for a in "xyz":
        p = os.path.join(D, split, "Inertial Signals", f"total_acc_{a}_{split}.txt")
        axes.append(np.loadtxt(p, dtype=np.float32))
    return np.stack(axes, axis=-1)  # (n, 128, 3)

labels = [l.split()[1].replace("_", " ").title()
          for l in open(os.path.join(D, "activity_labels.txt"))]
Xtr, ytr, str_ = load("train")
Xte, yte, ste = load("test")
print("train", Xtr.shape, "test", Xte.shape, labels)

# Hold out whole subjects from train for tuning (test set is never touched until the end)
subs = np.unique(str_)
val_subs = rng.choice(subs, 5, replace=False)
vmask = np.isin(str_, val_subs)
mu, sd = Xtr[~vmask].mean(0), Xtr[~vmask].std(0) + 1e-6
norm = lambda X: (X - mu) / sd

def init(h):
    return {"W1": rng.normal(0, np.sqrt(2 / 561), (561, h)).astype(np.float32), "b1": np.zeros(h, np.float32),
            "W2": rng.normal(0, np.sqrt(2 / h), (h, 6)).astype(np.float32), "b2": np.zeros(6, np.float32)}

def forward(P, X):
    H = np.maximum(0, X @ P["W1"] + P["b1"])
    Z = H @ P["W2"] + P["b2"]
    Z -= Z.max(1, keepdims=True)
    E = np.exp(Z)
    return H, E / E.sum(1, keepdims=True)

def train(X, y, h=64, epochs=40, lr=1e-3, wd=1e-4, drop=0.3, Xv=None, yv=None):
    P = init(h)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(x) for k, x in P.items()}
    t = 0
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        for i in range(0, len(X), 64):
            b = idx[i:i + 64]
            xb, yb = X[b], y[b]
            H = np.maximum(0, xb @ P["W1"] + P["b1"])
            mask = (rng.random(H.shape) > drop) / (1 - drop)
            Hd = H * mask
            Z = Hd @ P["W2"] + P["b2"]; Z -= Z.max(1, keepdims=True)
            E = np.exp(Z); pr = E / E.sum(1, keepdims=True)
            dZ = pr; dZ[np.arange(len(b)), yb] -= 1; dZ /= len(b)
            g = {"W2": Hd.T @ dZ + wd * P["W2"], "b2": dZ.sum(0)}
            dH = (dZ @ P["W2"].T) * mask * (H > 0)
            g["W1"] = xb.T @ dH + wd * P["W1"]; g["b1"] = dH.sum(0)
            t += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    return P

acc = lambda P, X, y: float((forward(P, X)[1].argmax(1) == y).mean())

# Small hyperparameter search on the validation subjects
best = None
for h in (32, 64, 128):
    for drop in (0.2, 0.5):
        P = train(norm(Xtr[~vmask]), ytr[~vmask], h=h, drop=drop)
        a = acc(P, norm(Xtr[vmask]), ytr[vmask])
        print(f"hidden={h} dropout={drop} val_acc={a:.4f}")
        if best is None or a > best[0]:
            best = (a, h, drop)
print("best", best)
_, H_, DROP = best

# Final model: train on all training subjects, evaluate once on test
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
P = train(norm(Xtr), ytr, h=H_, drop=DROP)
probs = forward(P, norm(Xte))[1]
pred = probs.argmax(1)
test_acc = float((pred == yte).mean())
base_label = int(np.bincount(ytr).argmax())
base_acc = float((yte == base_label).mean())
cm = np.zeros((6, 6), int)
for a, b in zip(yte, pred): cm[a, b] += 1
print(f"TEST acc={test_acc:.4f}  baseline('{labels[base_label]}')={base_acc:.4f}")
print(cm)

# Export model (rounded to keep file small) -- runs in browser JS
r = lambda a: np.round(a, 4).tolist()
json.dump({"mu": r(mu), "sd": r(sd), "W1": r(P["W1"]), "b1": r(P["b1"]),
           "W2": r(P["W2"]), "b2": r(P["b2"]), "labels": labels}, open(os.path.join(OUT, "model.json"), "w"))

# Gallery: raw traces + predictions. Mix of random correct and ALL-ish mistakes
sig = signals("test")
wrong = np.where(pred != yte)[0]
right = np.where(pred == yte)[0]
pick = list(wrong) + list(rng.choice(right, 120, replace=False))
rng.shuffle(pick)
gallery = [{"id": int(i), "subject": int(ste[i]), "true": int(yte[i]), "pred": int(pred[i]),
            "probs": r(probs[i]), "trace": np.round(sig[i], 3).T.tolist()} for i in pick]
json.dump(gallery, open(os.path.join(OUT, "gallery.json"), "w"))

# Mean trace per activity (for overlap comparison) + summary stats for the guide
json.dump({"labels": labels, "test_acc": test_acc, "baseline_acc": base_acc,
           "baseline_label": labels[base_label], "confusion": cm.tolist(),
           "hidden": H_, "dropout": DROP, "val_acc": best[0],
           "n_train": len(Xtr), "n_test": len(Xte), "n_wrong": int(len(wrong)),
           "val_subjects": [int(s) for s in val_subs]},
          open(os.path.join(OUT, "results.json"), "w"), indent=1)
print("gallery items:", len(gallery), "wrong:", len(wrong))
