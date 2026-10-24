"""M1 inference: load saved artifacts once, then `recommend()` is a plain function.

Stage 1: calibrated crop ranking (top-3), input-range guard, low-confidence flag.
Stage 2: fertilizer for the top-1 crop (notebook model, unchanged until Task 2).

Artifacts (models/): crop_config.json, crop_preprocessors.pkl, crop_model*.keras,
fert_model.keras, fert_preprocessors.pkl. Produced by
`scripts/task1_calibration.py final` and `scripts/train_stage2_notebook.py`.
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

        d = Path(models_dir)
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
        if (d / "fert_model.keras").exists():
            self.fert_model = keras.models.load_model(d / "fert_model.keras")
            with open(d / "fert_preprocessors.pkl", "rb") as fh:
                self.fert_pre = pickle.load(fh)

    @property
    def soil_types(self):
        return [] if self.fert_model is None else self.fert_pre["soil_encoder"].classes_.tolist()

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
                "provenance": provenance, "note": None}
        if tier is None:
            return base
        if self.fert_model is None:
            return {**base, "note": "Fertilizer model not loaded."}
        pre = self.fert_pre
        if soil_type not in set(pre["soil_encoder"].classes_):
            return {**base, "note": f"Unknown soil type '{soil_type}'."}
        xn = pre["scaler_fert"].transform([[temperature, humidity, moisture, N, K, P]])
        probs = np.asarray(self.fert_model.predict(
            [xn, pre["soil_encoder"].transform([soil_type]), pre["fert_crop_encoder"].transform([fert_crop])],
            verbose=0))[0]
        return {**base, "fertilizer": str(pre["fert_encoder"].classes_[probs.argmax()]),
                "confidence": float(probs.max()),
                "note": "Uncalibrated stage-2 confidence (notebook model; 99 training rows)."}

    # ---------------------------------------------------------------- public API
    def recommend(self, N, P, K, temperature, humidity, ph, rainfall, soil_type=None, moisture=None):
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
        return {"top3": top3, "status": status, "uncertain": bool(uncertain), "warnings": warnings,
                "message": message, "fertilizer": fert,
                "model": {"variant": self.cfg["variant"], "temperature": self.temperature,
                          "uncertain_threshold": self.uncertain_threshold}}


@lru_cache(maxsize=1)
def get_recommender(models_dir=str(DEFAULT_MODELS_DIR)):
    return M1Recommender(models_dir)


def recommend(N, P, K, temperature, humidity, ph, rainfall, soil_type=None, moisture=None):
    """Module-level entry point used by the Gradio app (loads the artifacts once)."""
    return get_recommender().recommend(N, P, K, temperature, humidity, ph, rainfall, soil_type, moisture)
