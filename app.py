from flask import Flask, request, render_template
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

app = Flask(__name__)

# Load model and tokenizer
model_name = "Kaludi/Reviews-Sentiment-Analysis"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        # sentiment = 'Positive' if probs[0][1] > probs[0][0] else 'Negative'
        if probs[0][1] > probs[0][0]:
            sentiment = probs[0][1].item()
        else:
            sentiment = -probs[0][0].item()
        return render_template("index.html", sentiment=sentiment, text=text)
    return render_template("index.html", sentiment="", text="")


if __name__ == "__main__":
    app.run(debug=True)
