# Aspect-Based Sentiment Analysis for Restaurant Reviews

This project demonstrates an end-to-end workflow for aspect-based sentiment analysis on restaurant reviews. It includes data preprocessing, model fine-tuning, inference, and a Gradio demo app prepared for deployment on Hugging Face Spaces.

## What This Project Does

The project is split into two main tasks:

- `ABTE` (Aspect-Based Term Extraction): extract aspect terms from a restaurant review.
- `ABSA` (Aspect-Based Sentiment Analysis): predict the sentiment associated with the review or each extracted aspect.

Together, these tasks allow the system to identify what a review is talking about and whether the sentiment is positive, neutral, or negative.

## Models

Two task-specific models were fine-tuned from the pretrained [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) backbone:

- ABTE model: fine-tuned for token classification to extract aspect terms from restaurant reviews
- ABSA model: fine-tuned for sequence classification to predict sentiment polarity for restaurant review content

The final trained checkpoints were published to Hugging Face Hub as:

- [GinNoV111/abte-restaurant-model](https://huggingface.co/GinNoV111/abte-restaurant-model)
- [GinNoV111/absa-restaurant-model](https://huggingface.co/GinNoV111/absa-restaurant-model)

## Dataset

The models were trained using the [thainq107/abte-restaurants](https://huggingface.co/datasets/thainq107/abte-restaurants) dataset from Hugging Face.

This dataset is designed for aspect-based sentiment analysis in the restaurant domain and contains review text annotated for aspect-related tasks. It supports:

- aspect term extraction for identifying important opinion targets in a sentence
- sentiment analysis for determining whether the opinion toward an aspect is positive, neutral, or negative

Using a restaurant-specific dataset helps the models focus on domain-relevant aspects such as food, service, price, drinks, and ambiance.

## Main Files

- [app.py](./app.py): Gradio demo app for Hugging Face Spaces
- [requirements.txt](./requirements.txt): dependencies for the app
- [Aspect-based Term Extraction.ipynb](<./Aspect-based Term Extraction.ipynb>): notebook for ABTE training and experiments
- [Aspect-based Sentiment Analysis.ipynb](<./Aspect-based Sentiment Analysis.ipynb>): notebook for ABSA training and experiments

## Deployment

The interactive demo is deployed on Hugging Face Spaces:

- Hugging Face Spaces: [GinNoV111/absa-restaurant-demo](https://huggingface.co/spaces/GinNoV111/absa-restaurant-demo)

## Output Example

Given a review such as:

```text
The food was delicious but the service was very slow.
```

The system can:

- extract aspects like `food` and `service`
- predict sentiment for each aspect
- show the overall sentiment of the review

## Summary

This project combines model training, inference, and deployment for restaurant review analysis using aspect-based NLP techniques. It is suitable for demonstration, experimentation, and sharing through a web interface.
