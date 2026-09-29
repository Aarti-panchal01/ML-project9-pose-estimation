"""Load the trained ANN and classify features. This is what the particle filter calls.

    from predict import PoseClassifier
    clf = PoseClassifier()
    label, pose = clf.predict(features)          # features: (42,) or (N, 42)
"""
import os

import joblib
import numpy as np

from simulator import label_to_pose

ROOT = os.path.join(os.path.dirname(__file__), "..")


class PoseClassifier:
    def __init__(self, path=os.path.join(ROOT, "models", "pose_ann.joblib")):
        bundle = joblib.load(path)
        self.model, self.scaler = bundle["model"], bundle["scaler"]

    def predict_proba(self, features):
        """Returns (N, 4800) probabilities over pose labels."""
        X = self.scaler.transform(np.atleast_2d(features))
        return self.model.predict_proba(X)

    def predict(self, features):
        """Returns (label, bin-centre pose [x, y, z, phi]). Scalar/1-D in, scalar/1-D out."""
        single = np.asarray(features).ndim == 1
        labels = self.model.classes_[self.predict_proba(features).argmax(1)]
        poses = label_to_pose(labels)
        return (labels[0], poses[0]) if single else (labels, poses)


def load_error_distribution(path=os.path.join(ROOT, "results", "error_distribution.npz")):
    """(M, 4) array of (true - classified) deltas in [x, y, z, phi]. Sample rows for particles."""
    return np.load(path)["delta_true"]
