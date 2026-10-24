"""Live demo: upload (or pick) a blood cell image and see the prediction, the class probabilities and a Grad-CAM heatmap.

    pip install gradio
    python demo/app.py --checkpoint <run>/best.pt            # local URL
    python demo/app.py --checkpoint <run>/best.pt --share    # public link (handy in Colab)

The final model is the SmallCNN chosen by validation, e.g. experiments_groupsplit/smallcnn_scratch_s42/best.pt.
If evaluate.py was run for that checkpoint, its test results (eval/metrics.json) are shown in the page header.
Research prototype, not a clinical tool. Example images come from `python demo/prepare_demo.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predictor import Predictor  # noqa: E402

MODEL_NAMES = {"smallcnn": "small CNN trained from scratch", "resnet50": "fine-tuned ResNet-50",
               "efficientnet_b0": "fine-tuned EfficientNet-B0", "densenet169": "fine-tuned DenseNet-169"}


def describe(predictor: Predictor, checkpoint: str) -> str:
    text = (f"A {MODEL_NAMES.get(predictor.model_name, predictor.model_name)} that names the white blood cell type in "
            "a microscope image (eosinophil, lymphocyte, monocyte, neutrophil). The heatmap (Grad-CAM) shows which "
            "regions pushed the network towards its answer.")
    metrics = Path(checkpoint).parent / "eval" / "metrics.json"
    if metrics.exists():
        m = json.loads(metrics.read_text())
        text += (f"\n\nThis checkpoint on the held-out test folder: accuracy {m['accuracy']:.1%}, "
                 f"macro-F1 {m['macro_f1']:.3f}")
        clean = m.get("clean_subset", {})
        if clean.get("n"):
            text += (f"; on the {clean['n']:,} test images whose source photo was never in training: "
                     f"accuracy {clean['accuracy']:.1%}, macro-F1 {clean['macro_f1']:.3f}")
        per_class = metrics.parent / "per_class.csv"
        if per_class.exists():
            import pandas as pd
            pc = pd.read_csv(per_class).sort_values("recall")
            text += ". Lowest recall: " + ", ".join(f"{r['class']} {r['recall']:.0%}" for _, r in pc.head(2).iterrows())
        text += "."
    text += ("\n\n**Research prototype, not a clinical tool.** Trained on the Kaggle Blood Cell Images data set "
             "(Mooney, 2018) with a split by source photo, so no photo is shared between training and validation.")
    return text


def build_demo(checkpoint: str, examples_dir: str):
    import gradio as gr

    predictor = Predictor(checkpoint)

    def classify(image):
        if image is None:
            return None, None
        r = predictor.predict(image)
        return r["probs"], r["overlay"]

    examples = sorted(str(p) for p in Path(examples_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Blood smear image (320×240 works best)"),
        outputs=[
            gr.Label(num_top_classes=len(predictor.classes), label="Predicted cell type"),
            gr.Image(label="Where the network looks (Grad-CAM)"),
        ],
        examples=[[e] for e in examples] or None,
        title="White blood cell classifier",
        description=describe(predictor, checkpoint),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--examples-dir", default="demo/examples")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    a = ap.parse_args(argv)
    build_demo(a.checkpoint, a.examples_dir).launch(share=a.share)


if __name__ == "__main__":
    main()
