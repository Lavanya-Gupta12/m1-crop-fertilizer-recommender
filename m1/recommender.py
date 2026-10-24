"""M1 inference: load saved artifacts once, then `recommend()` is a plain function.

Stage 1: calibrated crop ranking (top-3), input-range guard, low-confidence flag.
Stage 2: fertilizer for the top-1 crop via the tier mapping (m1/tiers.py), RandomForest
         pipeline, with its own input-range guard.

Artifacts (models/): crop_config.json, crop_preprocessors.pkl, crop_model*.keras,
fert_config.json, fert_rf.joblib. Produced by `scripts/task1_calibration.py final` and
`scripts/train_stage2.py`.
"""
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class M1Recommender:
    def __init__(self, models_dir=DEFAULT_MODELS_DIR, uncertain_threshold=None):
        import keras
        from m1.calibration import InputGuard

        d = self.models_dir = Path(models_dir)
        self.cfg = json.loads((d / "crop_config.json").read_text())
        with open(d / "crop_preprocessors.pkl", "rb") as fh:
            pre = pickle.load(fh)
        self.scaler = pre["scaler_crop"]
        self.classes = np.array(self.cfg["classes"])
        self.temperature = float(self.cfg["temperature"])
        # configurable; default is the threshold chosen from the out-of-fold coverage curve
        self.uncertain_threshold = (self.cfg["uncertain_threshold"] if uncertain_threshold is None
                                    else uncertain_threshold)
        g = self.cfg["guard"]
        self.guard = InputGuard(g["feature_names"], g["margin"])
        self.guard.lower_, self.guard.upper_ = np.array(g["lower"]), np.array(g["upper"])
        self.guard.train_min_, self.guard.train_max_ = np.array(g["train_min"]), np.array(g["train_max"])
        self.logit_models = []
        for name in self.cfg["model_files"]:
            m = keras.models.load_model(d / name)
            self.logit_models.append(keras.Model(m.inputs, m.get_layer("logits").output))

        self.fert_model = None
        if (d / "fert_config.json").exists():
            import joblib
            self.fert_cfg = json.loads((d / "fert_config.json").read_text())
            self.fert_model = joblib.load(d / self.fert_cfg["model_file"])
            fg = self.fert_cfg["guard"]
            self.fert_guard = InputGuard(fg["feature_names"], fg["margin"])
            self.fert_guard.lower_, self.fert_guard.upper_ = np.array(fg["lower"]), np.array(fg["upper"])
            self.fert_guard.train_min_, self.fert_guard.train_max_ = np.array(fg["train_min"]), np.array(fg["train_max"])

    @property
    def soil_types(self):
        return [] if self.fert_model is None else list(self.fert_cfg["soil_types"])

    def fert_frame(self, temperature, humidity, moisture, N, K, P, soil_type, fert_crop):
        import pandas as pd
        row = dict(zip(self.fert_cfg["num_cols"], [temperature, humidity, moisture, N, K, P]))
        row.update({"soil_type": soil_type, "label": fert_crop})
        return pd.DataFrame([row])[self.fert_cfg["num_cols"] + self.fert_cfg["cat_cols"]]

    # ---------------------------------------------------------------- stage 1
    def crop_probs(self, x_raw):
        """Calibrated class probabilities for one or more raw input rows (N, P, K, temp, hum, pH, rain)."""
        import keras
        x = self.scaler.transform(np.atleast_2d(np.asarray(x_raw, dtype=float)))
        # direct call instead of .predict(): no per-call retracing, much faster for single rows
        logits = [np.asarray(keras.ops.convert_to_numpy(m([x], training=False)), dtype=np.float64)
                  for m in self.logit_models]
        if self.cfg["aggregate"] == "mean_prob":
            z = np.log(np.clip(np.mean([_softmax(l) for l in logits], axis=0), 1e-12, 1.0))
        else:
            z = logits[0]
        return _softmax(z / self.temperature)

    # ---------------------------------------------------------------- stage 2
    def fertilizer(self, crop, temperature, humidity, moisture, N, K, P, soil_type):
        from m1 import tiers
        tier, fert_crop, provenance = tiers.resolve(crop)
        base = {"fertilizer": None, "confidence": None, "tier": tier, "fert_crop": fert_crop,
                "provenance": provenance, "warnings": [], "note": None}
        if tier is None:
            return base
        if self.fert_model is None:
            return {**base, "note": "Fertilizer model not loaded."}
        if soil_type not in set(self.fert_cfg["soil_types"]):
            return {**base, "note": f"Unknown soil type '{soil_type}'."}
        probs = self.fert_model.predict_proba(
            self.fert_frame(temperature, humidity, moisture, N, K, P, soil_type, fert_crop))[0]
        return {**base, "fertilizer": str(self.fert_model.classes_[probs.argmax()]),
                "confidence": float(probs.max()),
                "warnings": self.fert_guard.check([temperature, humidity, moisture, N, K, P]),
                "note": ("Confidence is uncalibrated (RandomForest on 99 rows; under-confident in CV: "
                         "accuracy 0.97 vs mean confidence 0.64).")}

    # ---------------------------------------------------------------- explanations
    @staticmethod
    def _format_reason(names, values, label):
        order = np.argsort(-np.abs(values))[:3]
        parts = ", ".join(f"{names[i]} {values[i]:+.2f}" for i in order)
        return f"Top drivers for '{label}' (SHAP, change in probability vs. a typical input): {parts}"

    def crop_reason(self, x_raw, class_idx):
        """Kernel SHAP for one input and one class of the deployed calibrated ensemble."""
        import shap
        if not hasattr(self, "_crop_bg"):
            self._crop_bg = np.load(self.models_dir / "shap_background.npz")["crop_background"]
        f = lambda Z: self.crop_probs(Z)[:, class_idx]                       # noqa: E731
        sv = shap.KernelExplainer(f, self._crop_bg).shap_values(
            np.atleast_2d(np.asarray(x_raw, dtype=float)), nsamples=128, silent=True)
        return self._format_reason(self.cfg["feature_names"], np.asarray(sv)[0], self.classes[class_idx])

    def fert_reason(self, frame, fert_label):
        """Exact TreeSHAP for the fertilizer RandomForest; one-hot columns summed per input."""
        import shap
        pre, clf = self.fert_model.named_steps["pre"], self.fert_model.named_steps["clf"]
        if not hasattr(self, "_fert_explainer"):
            self._fert_explainer = shap.TreeExplainer(clf)
        Xt = pre.transform(frame)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
        k = list(clf.classes_).index(fert_label)
        sv = np.asarray(self._fert_explainer.shap_values(Xt))[0, :, k]
        cols = pre.get_feature_names_out()
        names = self.fert_cfg["num_cols"] + ["soil_type", "crop"]
        vals = [sv[[i for i, c in enumerate(cols) if c == f"num__{n}"]].sum() for n in self.fert_cfg["num_cols"]]
        vals += [sv[[i for i, c in enumerate(cols) if c.startswith("cat__soil_type_")]].sum(),
                 sv[[i for i, c in enumerate(cols) if c.startswith("cat__label_")]].sum()]
        return self._format_reason(names, np.array(vals), fert_label)

    # ---------------------------------------------------------------- public API
    def recommend(self, N, P, K, temperature, humidity, ph, rainfall, soil_type=None, moisture=None,
                  explain=True):
        x = [N, P, K, temperature, humidity, ph, rainfall]
        probs = self.crop_probs(x)[0]
        order = np.argsort(probs)[::-1][:3]
        top3 = [{"crop": str(self.classes[i]), "probability": float(probs[i])} for i in order]
        warnings = self.guard.check(x)
        uncertain = top3[0]["probability"] < self.uncertain_threshold
        if warnings:
            status = "out_of_range"
            message = ("Input outside the range the model was trained on; the ranking below is "
                       "not reliable regardless of the probabilities shown.")
        elif uncertain:
            status = "uncertain"
            message = (f"Uncertain: top crop probability {top3[0]['probability']:.0%} is below "
                       f"{self.uncertain_threshold:.0%}. Consider all three options.")
        else:
            status = "ok"
            message = None
        fert = (self.fertilizer(top3[0]["crop"], temperature, humidity, moisture, N, K, P, soil_type)
                if soil_type is not None and moisture is not None else None)
        reasons = {"crop": None, "fertilizer": None}
        if explain:
            reasons["crop"] = self.crop_reason(x, int(order[0]))
            if fert and fert["fertilizer"] is not None:
                reasons["fertilizer"] = self.fert_reason(
                    self.fert_frame(temperature, humidity, moisture, N, K, P, soil_type, fert["fert_crop"]),
                    fert["fertilizer"])
        return {"top3": top3, "status": status, "uncertain": bool(uncertain), "warnings": warnings,
                "message": message, "fertilizer": fert, "reasons": reasons,
                "model": {"variant": self.cfg["variant"], "temperature": self.temperature,
                          "uncertain_threshold": self.uncertain_threshold}}


@lru_cache(maxsize=1)
def get_recommender(models_dir=str(DEFAULT_MODELS_DIR)):
    return M1Recommender(models_dir)


def recommend(N, P, K, temperature, humidity, ph, rainfall, soil_type=None, moisture=None, explain=True):
    """Module-level entry point used by the Gradio app (loads the artifacts once)."""
    return get_recommender().recommend(N, P, K, temperature, humidity, ph, rainfall, soil_type, moisture,
                                       explain)
