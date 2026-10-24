"""Calibration metrics, temperature scaling and an input-range guard for M1.

Everything here works on plain numpy arrays (probabilities or logits), so it is
independent of the Keras backend.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

EPS = 1e-12


# ---------------------------------------------------------------- metrics
def nll(probs, y):
    """Mean negative log-likelihood of the true class."""
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1.0))))


def brier(probs, y):
    """Multiclass Brier score: mean over samples of sum_k (p_k - onehot_k)^2. Range [0, 2]."""
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_bins(probs, y, n_bins=15):
    """Top-label reliability table with equal-width confidence bins.

    Returns a dict of per-bin arrays: lower/upper edge, count, mean confidence, accuracy.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins so that conf == 1.0 lands in the last bin
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        mean_conf = np.bincount(idx, weights=conf, minlength=n_bins) / count
        acc = np.bincount(idx, weights=correct, minlength=n_bins) / count
    return {"lower": edges[:-1], "upper": edges[1:], "count": count,
            "confidence": mean_conf, "accuracy": acc}


def ece(probs, y, n_bins=15):
    """Expected calibration error (top-label, equal-width bins)."""
    b = reliability_bins(probs, y, n_bins)
    mask = b["count"] > 0
    gaps = np.abs(b["accuracy"][mask] - b["confidence"][mask])
    return float(np.sum(b["count"][mask] * gaps) / len(y))


def all_metrics(probs, y, n_bins=15):
    from sklearn.metrics import accuracy_score, f1_score
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ece": ece(probs, y, n_bins),
        "nll": nll(probs, y),
        "brier": brier(probs, y),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }


def plot_reliability(probs, y, ax, title="", n_bins=15):
    b = reliability_bins(probs, y, n_bins)
    mask = b["count"] > 0
    centers = (b["lower"] + b["upper"]) / 2
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", label="perfect calibration")
    ax.bar(centers[mask], b["accuracy"][mask], width=1.0 / n_bins, edgecolor="black",
           alpha=0.7, label="accuracy in bin")
    ax.plot(b["confidence"][mask], b["accuracy"][mask], "o", color="C3", ms=4,
            label="(mean conf, acc)")
    for c, a, n in zip(centers[mask], b["accuracy"][mask], b["count"][mask]):
        ax.annotate(str(n), (c, min(a, 0.97)), ha="center", va="bottom", fontsize=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("confidence (top-1 probability)"); ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE={ece(probs, y, n_bins):.4f}")
    ax.legend(loc="upper left", fontsize=7)


# ---------------------------------------------------------- temperature scaling
class TemperatureScaler:
    """Single-parameter temperature scaling (Guo et al., 2017).

    Fit on a held-out validation split's logits only; never on test data.
    """

    def __init__(self, bounds=(0.05, 20.0)):
        self.bounds = bounds
        self.temperature = 1.0

    def fit(self, logits, y):
        def loss(log_t):
            lp = log_softmax(logits / np.exp(log_t), axis=1)
            return -np.mean(lp[np.arange(len(y)), y])

        res = minimize_scalar(loss, bounds=np.log(self.bounds), method="bounded")
        self.temperature = float(np.exp(res.x))
        return self

    def transform(self, logits):
        return softmax(logits / self.temperature, axis=1)


# ------------------------------------------------------------- input guard
class InputGuard:
    """Flags inputs outside the training range (min/max widened by `margin` x range)."""

    def __init__(self, feature_names, margin=0.10):
        self.feature_names = list(feature_names)
        self.margin = margin

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        lo, hi = X.min(axis=0), X.max(axis=0)
        pad = self.margin * (hi - lo)
        self.lower_, self.upper_ = lo - pad, hi + pad
        self.train_min_, self.train_max_ = lo, hi
        return self

    def check(self, x):
        """Return a list of human-readable warnings for one input row (empty if in range)."""
        x = np.asarray(x, dtype=float).ravel()
        warnings = []
        for name, v, lo, hi, tmin, tmax in zip(self.feature_names, x, self.lower_, self.upper_,
                                                self.train_min_, self.train_max_):
            if v < lo or v > hi:
                warnings.append(f"{name}={v:g} is outside the training range "
                                f"[{tmin:g}, {tmax:g}]")
        return warnings

    def to_dict(self):
        return {"feature_names": self.feature_names, "margin": self.margin,
                "lower": self.lower_.tolist(), "upper": self.upper_.tolist(),
                "train_min": self.train_min_.tolist(), "train_max": self.train_max_.tolist()}
