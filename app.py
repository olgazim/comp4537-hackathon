from flask import Flask, request, render_template
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch.nn.functional as F

app = Flask(__name__)

# Load sentiment analysis model and tokenizer
sent_model = AutoModelForSequenceClassification.from_pretrained(
    "Kaludi/Reviews-Sentiment-Analysis"
)
sent_tokenizer = AutoTokenizer.from_pretrained("Kaludi/Reviews-Sentiment-Analysis")

# Load paraphrasing model and tokenizer
para_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
para_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]

        # Sentiment Analysis
        inputs = sent_tokenizer(text, return_tensors="pt")
        outputs = sent_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        negative_score, positive_score = probs[0][0].item(), probs[0][1].item()

        # Paraphrasing
        paraphrase_input = f"paraphrase: {text} </s>"
        encoding = para_tokenizer(
            paraphrase_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        para_outputs = para_model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=256,
            num_beams=5,
            num_return_sequences=3,
            early_stopping=True,
        )
        paraphrases = [
            para_tokenizer.decode(output, skip_special_tokens=True)
            for output in para_outputs
        ]

        return render_template(
            "index.html",
            negative_score=negative_score,
            positive_score=positive_score,
            original_text=text,
            paraphrases=paraphrases,
        )
    return render_template(
        "index.html",
        negative_score=None,
        positive_score=None,
        original_text="",
        paraphrases=[],
    )


if __name__ == "__main__":
    app.run(debug=True)
