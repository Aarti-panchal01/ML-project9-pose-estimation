"""Train the pose classifier ANN (paper Section IV-D).

Architecture: 42 inputs -> 100 ReLU -> 100 ReLU -> 4800-way softmax, Adam optimizer.
Implemented with scikit-learn's MLPClassifier, trained one epoch at a time so we can
log validation accuracy per epoch, checkpoint the best model, and stop early.

Usage:
    python src/train.py --epochs 40            # trains, stops early if val accuracy stalls
    python src/train.py --resume --epochs 10   # continue from the checkpoint for 10 more epochs
    python src/train.py --eval-only            # recompute metrics from the saved model
Outputs:
    models/pose_ann.joblib          best model + input scaler (saved whenever val acc improves)
    results/training_log.csv        per-epoch loss / val accuracy (written as it goes)
    results/metrics.json            final accuracy numbers
    results/error_distribution.npz  val-set errors, used by the particle filter
"""
import argparse
import copy
import json
import os
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import config as C
from simulator import label_to_pose

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(ROOT, "models", "pose_ann.joblib")
LOG_PATH = os.path.join(ROOT, "results", "training_log.csv")
CHUNK = 8192   # sklearn one-hot encodes y per call; 4800 classes is memory-heavy, so feed chunks


def predict_chunked(model, X, topk=1):
    """Argmax (and top-k) predictions without holding the full (N, 4800) matrix in memory."""
    pred, top = [], []
    for s in range(0, len(X), CHUNK):
        p = model.predict_proba(X[s:s + CHUNK])
        pred.append(model.classes_[p.argmax(1)])
        top.append(model.classes_[np.argsort(p, axis=1)[:, -topk:]])
    return np.concatenate(pred), np.concatenate(top)


def bin_distance(a, b):
    """Max per-dimension bin index difference between two label arrays."""
    shape = tuple(C.BINS[k][2] for k in ("x", "y", "z", "phi"))
    ia = np.stack(np.unravel_index(a, shape), axis=-1)
    ib = np.stack(np.unravel_index(b, shape), axis=-1)
    return np.abs(ia - ib).max(axis=-1)


def evaluate(model, Xva, yva, poses_va):
    """Final metrics + error distributions (paper Fig 4 and Fig 5) for the particle filter."""
    pred, top5 = predict_chunked(model, Xva, topk=5)
    metrics = {
        "val_accuracy": float((pred == yva).mean()),
        "val_top5_accuracy": float((top5 == yva[:, None]).any(1).mean()),
        "val_within_1_bin_every_dim": float((bin_distance(pred, yva) <= 1).mean()),
        "paper_val_accuracy": 0.54,
        "n_val": int(len(yva)),
    }
    classified = label_to_pose(pred)                   # bin centre of the predicted label
    np.savez(os.path.join(ROOT, "results", "error_distribution.npz"),
             delta_true=poses_va[:, :4] - classified,            # true - classified (particle filter)
             delta_label=label_to_pose(yva) - classified,        # correct bin - predicted bin (Fig 5)
             columns=np.array(["x", "y", "z", "phi"]))
    with open(os.path.join(ROOT, "results", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "dataset.npz"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4, help="stop after this many epochs with no gain")
    ap.add_argument("--resume", action="store_true", help="continue training from models/pose_ann.joblib")
    ap.add_argument("--no-eval", action="store_true", help="skip the final evaluation (use with --resume)")
    ap.add_argument("--eval-only", action="store_true", help="skip training, evaluate the saved model")
    args = ap.parse_args()

    d = np.load(args.data)
    X, y, poses = d["X"], d["y"], d["poses"]
    tr, va = d["train_idx"], d["val_idx"]
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    if args.eval_only:
        bundle = joblib.load(MODEL_PATH)
        evaluate(bundle["model"], bundle["scaler"].transform(X[va]), y[va], poses[va])
        return

    classes = np.arange(C.N_LABELS)
    if args.resume:
        bundle = joblib.load(MODEL_PATH)
        model, scaler = bundle["model"], bundle["scaler"]
        log = np.atleast_1d(np.genfromtxt(LOG_PATH, delimiter=",", names=True))
        start = int(log["epoch"].max()) + 1
        best_acc, best_model = float(log["val_accuracy"].max()), copy.deepcopy(model)
        print(f"resuming from epoch {start}, best val acc so far {best_acc:.4f}")
    else:
        scaler = StandardScaler().fit(X[tr])
        model = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu", solver="adam",
                              learning_rate_init=args.lr, batch_size=args.batch, random_state=C.SEED)
        with open(LOG_PATH, "w") as f:
            f.write("epoch,train_loss,val_accuracy\n")
        start, best_acc, best_model = 1, -1.0, None
    Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
    ytr, yva = y[tr], y[va]
    rng = np.random.default_rng(C.SEED + start)
    stale = 0
    t0 = time.time()

    for epoch in range(start, start + args.epochs):
        order = rng.permutation(len(Xtr))
        losses = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for s in range(0, len(order), CHUNK):
                idx = order[s:s + CHUNK]
                model.partial_fit(Xtr[idx], ytr[idx], classes=classes)
                losses.append(model.loss_)
        loss = float(np.mean(losses))
        val_acc = float((predict_chunked(model, Xva)[0] == yva).mean())
        print(f"epoch {epoch:3d}  train loss {loss:.4f}  val acc {val_acc:.4f}  ({time.time() - t0:.0f}s)")

        with open(LOG_PATH, "a") as f:
            f.write(f"{epoch},{loss:.5f},{val_acc:.5f}\n")
        if val_acc > best_acc + 1e-3:
            best_acc, best_model, stale = val_acc, copy.deepcopy(model), 0
            joblib.dump({"model": best_model, "scaler": scaler}, MODEL_PATH)      # checkpoint
        else:
            stale += 1
            if stale >= args.patience:
                print(f"no improvement for {args.patience} epochs, stopping early")
                break

    if not args.no_eval:
        evaluate(best_model, Xva, yva, poses[va])


if __name__ == "__main__":
    main()
