"""M1 Gradio app: thin UI over m1.recommender.recommend() (loads saved artifacts, no retraining).

Usage:
    python app.py              # local, http://127.0.0.1:7860
    python app.py --share      # temporary public link (about 1 week)
    python app.py --host 0.0.0.0 --port 7860   # for embedding in the team HTML hub
"""
import argparse

import gradio as gr

from m1.recommender import get_recommender, recommend

INPUT_LABELS = ["Nitrogen (N)", "Phosphorous (P)", "Potassium (K)", "Temperature (°C)", "Humidity (%)",
                "pH", "Rainfall (mm)", "Soil Type", "Moisture (%)"]

# clickable example rows (same order as INPUT_LABELS)
EXAMPLES = [
    [90, 42, 43, 20.8, 82, 6.5, 202, "Clayey", 45],
    [40, 68, 80, 18.9, 17, 7.3, 80, "Loamy", 40],
    [90, 42, 43, 20.8, 82, 14, 202, "Clayey", 45],
    # an actual mango row from the crop dataset (row index 1100); soil type and moisture are not in that data
    [2, 40, 27, 29.73770045, 47.54885174, 5.954626604, 90.09586854, "Loamy", 40],
]
EXAMPLE_LABELS = ["Rice-like", "Pulses (chickpea-like)", "Out-of-range pH",
                  "No fertilizer data (mango, real data row)"]


def gradio_predict(N, P, K, temperature, humidity, ph, rainfall, soil_type, moisture):
    values = [N, P, K, temperature, humidity, ph, rainfall, soil_type, moisture]
    missing = [label for label, v in zip(INPUT_LABELS, values) if v is None or v == ""]
    if missing:
        return (None, "**Please fill all inputs** before clicking Submit (or click one of the examples "
                      "below).\n\nMissing: " + ", ".join(missing), "", "")
    out = recommend(N, P, K, temperature, humidity, ph, rainfall, soil_type, moisture)
    crops = {t["crop"]: t["probability"] for t in out["top3"]}

    if out["status"] == "ok":
        status = "**Status:** OK"
    elif out["status"] == "uncertain":
        status = f"**Status: UNCERTAIN.** {out['message']}"
    else:
        status = f"**Status: INPUT OUT OF RANGE.** {out['message']}\n\n" + \
                 "\n".join(f"- {w}" for w in out["warnings"])
    status += (f"\n\n<sub>Model: {out['model']['variant']} (5-seed ensemble, temperature-scaled, "
               f"T={out['model']['temperature']:.2f}); uncertain below "
               f"{out['model']['uncertain_threshold']:.0%}.</sub>")

    f = out["fertilizer"]
    if f is None:
        fert_md = "Enter soil type and moisture for a fertilizer recommendation."
    else:
        tier = {1: "Tier 1 (ML model)", 2: "Tier 2 (crop-group proxy)"}.get(f["tier"], "No data")
        if f["fertilizer"] is None:
            fert_md = f"**{tier}** for {out['top3'][0]['crop']}. " + " ".join(s for s in (f["provenance"], f["note"]) if s)
        else:
            fert_md = (f"**{f['fertilizer']}** for {out['top3'][0]['crop']} ({f['confidence']:.0%})  \n"
                       f"**Source: {tier}.** {f['provenance']}  \n<sub>{f['note']}</sub>")
            if f["warnings"]:
                fert_md += ("\n\n**Fertilizer inputs outside the fertilizer data's range** "
                            "(the answer is an extrapolation):\n" + "\n".join(f"- {w}" for w in f["warnings"]))

    r = out["reasons"]
    reason_md = "\n\n".join(f"**{k.capitalize()}:** {v}" for k, v in (("crop", r["crop"]), ("fertilizer", r["fertilizer"])) if v)
    return crops, status, fert_md, reason_md


def build_demo():
    rec = get_recommender()
    return gr.Interface(
        fn=gradio_predict,
        inputs=[gr.Number(label=label, value=None) for label in INPUT_LABELS[:7]]
               + [gr.Dropdown(choices=rec.soil_types, label=INPUT_LABELS[7], value=None),
                  gr.Number(label=INPUT_LABELS[8], value=None)],
        outputs=[gr.Label(num_top_classes=3, label="Top-3 crops (calibrated probability)"),
                 gr.Markdown(label="Status"),
                 gr.Markdown(label="Recommended fertilizer"),
                 gr.Markdown(label="Why (SHAP)")],
        title="M1 — Crop & Fertilizer Recommender",
        examples=EXAMPLES,
        example_labels=EXAMPLE_LABELS,
        cache_examples=False,
        flagging_mode="never",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_demo().launch(share=a.share, server_name=a.host, server_port=a.port)
