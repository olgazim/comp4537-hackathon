---
tags:
- text-classification
language:
- en
widget:
- text: I don't feel like you trust me to do my job.
  example_title: "Negative Example 1"
- text: "This service was honestly one of the best I've experienced, I'll definitely come back!"
  example_title: "Positive Example 1"
- text: "I was extremely disappointed with this product. The quality was terrible and it broke after only a few days of use. Customer service was unhelpful and unresponsive. I would not recommend this product to anyone."
  example_title: "Negative Example 2"
- text: "I am so impressed with this product! The quality is outstanding and it has exceeded all of my expectations. The customer service team was also incredibly helpful and responsive to any questions I had. I highly recommend this product to anyone in need of a top-notch, reliable solution."
  example_title: "Positive Example 2"
datasets:
- Kaludi/data-reviews-sentiment-analysis
co2_eq_emissions:
  emissions: 24.76716845191504
---

# Reviews Sentiment Analysis

A tool that analyzes the overall sentiment of customer reviews for a specific product or service, whether it’s positive or negative. This analysis is performed by using natural language processing algorithms and machine learning from the model ‘Reviews-Sentiment-Analysis’ trained by Kaludi, allowing businesses to gain valuable insights into customer satisfaction and improve their products and services accordingly.

## Training Procedure

- learning_rate = 1e-5
- batch_size = 32
- warmup = 600
- max_seq_length = 128
- num_train_epochs = 10.0

## Validation Metrics

- Loss: 0.159
- Accuracy: 0.952
- Precision: 0.965
- Recall: 0.938
- AUC: 0.988
- F1: 0.951

## Usage

You can use cURL to access this model:

```
$ curl -X POST -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" -d '{"inputs": "I don't feel like you trust me to do my job."}' https://api-inference.huggingface.co/models/Kaludi/Reviews-Sentiment-Analysis
```

Or Python API:

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("Kaludi/Reviews-Sentiment-Analysis", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("Kaludi/Reviews-Sentiment-Analysis", use_auth_token=True)

inputs = tokenizer("I don't feel like you trust me to do my job.", return_tensors="pt")

outputs = model(**inputs)
```