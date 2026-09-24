"""Live demo: upload (or pick) a blood cell image and see the prediction, the class probabilities and a Grad-CAM heatmap.

    pip install gradio
    python demo/app.py --checkpoint outputs/resnet50_v1/best.pt            # local URL
    python demo/app.py --checkpoint outputs/resnet50_v1/best.pt --share    # public link (handy in Colab)

Research prototype, not a clinical tool. Example images come from `python demo/prepare_demo.py` (validation split).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predictor import Predictor  # noqa: E402

DESCRIPTION = (
    "Fine-tuned ResNet-50 that names the white blood cell type in a microscope image "
    "(eosinophil, lymphocyte, monocyte, neutrophil). The heatmap (Grad-CAM) shows which regions pushed the network "
    "towards its answer. **Research prototype, not a clinical tool.** Trained on the Kaggle Blood Cell Images data set."
)


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
        description=DESCRIPTION,
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
