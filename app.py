"""M1 Gradio app: thin UI over m1.recommender.recommend() (loads saved artifacts, no retraining).

Usage:
    python app.py              # local, http://127.0.0.1:7860
    python app.py --share      # temporary public link (about 1 week)
    python app.py --host 0.0.0.0 --port 7860   # for embedding in the team HTML hub
"""
import argparse

import gradio as gr

from m1.recommender import get_recommender, recommend


def gradio_predict(N, P, K, temperature, humidity, ph, rainfall, soil_type, moisture):
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
        fert_text = "Enter soil type and moisture for a fertilizer recommendation."
    elif f["fertilizer"] is None:
        fert_text = " ".join(s for s in (f["provenance"], f["note"]) if s)
    else:
        fert_text = (f"{f['fertilizer']} ({f['confidence']:.1%}, for {out['top3'][0]['crop']}). "
                     f"{f['provenance']} {f['note']}")
    return crops, status, fert_text


def build_demo():
    rec = get_recommender()
    return gr.Interface(
        fn=gradio_predict,
        inputs=[
            gr.Number(label="Nitrogen (N)", value=90),
            gr.Number(label="Phosphorous (P)", value=42),
            gr.Number(label="Potassium (K)", value=43),
            gr.Number(label="Temperature (°C)", value=20.8),
            gr.Number(label="Humidity (%)", value=82),
            gr.Number(label="pH", value=6.5),
            gr.Number(label="Rainfall (mm)", value=202),
            gr.Dropdown(choices=rec.soil_types, label="Soil Type", value="Clayey"),
            gr.Number(label="Moisture (%)", value=45),
        ],
        outputs=[gr.Label(num_top_classes=3, label="Top-3 crops (calibrated probability)"),
                 gr.Markdown(label="Status"),
                 gr.Textbox(label="Recommended Fertilizer")],
        title="M1 — Crop & Fertilizer Recommender",
        flagging_mode="never",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_demo().launch(share=a.share, server_name=a.host, server_port=a.port)
