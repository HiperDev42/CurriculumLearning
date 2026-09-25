from pathlib import Path
from typing import Any

import gradio as gr

from curriculum_learning.analysis import predict_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"


def list_models() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        path.name
        for path in OUTPUT_DIR.iterdir()
        if path.is_dir()
        and (
            (path / "config.json").exists()
            or (path / "trainer_state.json").exists()
            or any(path.glob("checkpoint-*/trainer_state.json"))
        )
    )


def classify(
    model_name: str, premise: str, hypothesis: str
) -> tuple[str, dict[str, float]]:
    if not model_name:
        raise gr.Error("Select a trained model first.")
    try:
        return predict_text(OUTPUT_DIR / model_name, premise, hypothesis)
    except ValueError as error:
        raise gr.Error(str(error)) from error


def build_demo() -> gr.Blocks:
    models = list_models()
    with gr.Blocks(title="Curriculum Learning Model Tester") as demo:
        gr.Markdown("# Curriculum Learning Model Tester")
        gr.Markdown(
            "Test a trained ASSIN2 entailment model with a premise and hypothesis."
        )
        model = gr.Dropdown(
            choices=models,
            value=models[0] if models else None,
            label="Trained model",
            info="The best validation checkpoint is loaded automatically.",
        )
        with gr.Row():
            premise = gr.Textbox(
                label="Premise", lines=5, placeholder="Write the premise..."
            )
            hypothesis = gr.Textbox(
                label="Hypothesis", lines=5, placeholder="Write the hypothesis..."
            )
        run = gr.Button("Classify", variant="primary")
        prediction = gr.Textbox(label="Prediction", interactive=False)
        probabilities = gr.Label(label="Class probabilities")
        run.click(
            classify,
            inputs=[model, premise, hypothesis],
            outputs=[prediction, probabilities],
        )
    return demo


def main() -> Any:
    build_demo().launch()


if __name__ == "__main__":
    main()
