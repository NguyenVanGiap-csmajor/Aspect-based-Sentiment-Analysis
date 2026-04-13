from __future__ import annotations

from functools import lru_cache
from typing import Any

import gradio as gr
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)


ABTE_MODEL_ID = "GinNoV111/abte-restaurant-model"
ABSA_MODEL_ID = "GinNoV111/absa-restaurant-model"
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]


@lru_cache(maxsize=1)
def load_resources() -> dict[str, Any]:
    abte_tokenizer = AutoTokenizer.from_pretrained(ABTE_MODEL_ID)
    abte_model = AutoModelForTokenClassification.from_pretrained(ABTE_MODEL_ID)
    abte_pipeline = pipeline(
        "token-classification",
        model=abte_model,
        tokenizer=abte_tokenizer,
        aggregation_strategy="simple",
    )

    absa_tokenizer = AutoTokenizer.from_pretrained(ABSA_MODEL_ID)
    absa_model = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL_ID)
    absa_model.eval()

    return {
        "abte_pipeline": abte_pipeline,
        "absa_tokenizer": absa_tokenizer,
        "absa_model": absa_model,
    }


def normalize_term(term: str) -> str:
    cleaned = term.replace(" ##", "").replace("##", "")
    return " ".join(cleaned.split()).strip()


def extract_aspects(text: str) -> list[dict[str, Any]]:
    predictions = load_resources()["abte_pipeline"](text)
    aspects: list[dict[str, Any]] = []
    seen: set[str] = set()

    for pred in predictions:
        raw_term = pred.get("word") or pred.get("entity_group") or ""
        term = normalize_term(raw_term)
        if not term:
            continue

        key = term.lower()
        if key in seen:
            continue

        seen.add(key)
        aspects.append(
            {
                "aspect": term,
                "confidence": round(float(pred.get("score", 0.0)), 4),
                "start": pred.get("start"),
                "end": pred.get("end"),
            }
        )

    return aspects


def predict_sentiment(text: str, aspect: str | None = None) -> dict[str, Any]:
    resources = load_resources()
    tokenizer = resources["absa_tokenizer"]
    model = resources["absa_model"]

    if aspect:
        inputs = tokenizer(
            text,
            aspect,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    best_idx = int(torch.argmax(probabilities).item())
    scores = {
        SENTIMENT_LABELS[idx]: round(float(probabilities[idx].item()), 4)
        for idx in range(len(SENTIMENT_LABELS))
    }

    return {
        "label": SENTIMENT_LABELS[best_idx],
        "score": scores[SENTIMENT_LABELS[best_idx]],
        "probabilities": scores,
    }


def analyze_text(text: str):
    text = text.strip()
    if not text:
        raise gr.Error("Please enter a sentence or review to analyze.")

    aspects = extract_aspects(text)
    overall_sentiment = predict_sentiment(text)

    rows = []
    for item in aspects:
        sentiment = predict_sentiment(text, item["aspect"])
        rows.append(
            [
                item["aspect"],
                item["confidence"],
                sentiment["label"],
                sentiment["score"],
            ]
        )

    if not rows:
        rows = [[
            "No aspect found",
            None,
            overall_sentiment["label"],
            overall_sentiment["score"],
        ]]

    summary = (
        f"{overall_sentiment['label']} "
        f"(Score: {overall_sentiment['score']:.4f})"
    )

    return summary, rows, overall_sentiment["probabilities"], aspects


with gr.Blocks(title="ABTE + ABSA Restaurant Demo") as demo:
    gr.Markdown(
        """
        # Demo Aspect-Based Sentiment Analysis
        Enter an English restaurant review. The system will extract aspect terms
        and predict the sentiment for each aspect.
        """
    )

    text_input = gr.Textbox(
        label="Enter a review",
        lines=5,
        placeholder="Example: The food was delicious but the service was very slow.",
    )

    gr.Examples(
        examples=[
            ["The food was delicious but the service was very slow."],
            ["The staff were friendly, but the price was too high."],
        ],
        inputs=text_input,
    )

    with gr.Row():
        submit_btn = gr.Button("Analyze", variant="primary")
        clear_btn = gr.Button("Clear")

    summary_output = gr.Textbox(label="Overall Sentiment", lines=1)
    aspect_table = gr.Dataframe(
        headers=["Aspect", "ABTE Score", "Sentiment", "ABSA Score"],
        label="Aspect-level results",
        wrap=True,
    )

    with gr.Row():
        probabilities_output = gr.JSON(label="Overall sentiment probabilities")
        raw_aspects_output = gr.JSON(label="Raw aspect output")

    submit_btn.click(
        fn=analyze_text,
        inputs=text_input,
        outputs=[
            summary_output,
            aspect_table,
            probabilities_output,
            raw_aspects_output,
        ],
    )

    clear_btn.click(
        fn=lambda: ("", [], {}, []),
        inputs=None,
        outputs=[
            summary_output,
            aspect_table,
            probabilities_output,
            raw_aspects_output,
        ],
    )

if __name__ == "__main__":
    demo.launch()
