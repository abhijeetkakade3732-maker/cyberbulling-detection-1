import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "cyberbullying_dataset.csv")


def train_model():
    """Load the supplied dataset and train a simple text classifier."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "cyberbullying_dataset.csv was not found. Keep it beside app.py."
        )

    data = pd.read_csv(DATASET_PATH)
    data.columns = [str(column).strip() for column in data.columns]

    if not {"Text", "Label"}.issubset(data.columns):
        raise ValueError("Dataset must contain columns named 'Text' and 'Label'.")

    data = data.dropna(subset=["Text", "Label"]).copy()
    data["Text"] = data["Text"].astype(str).str.strip()
    data["Label"] = data["Label"].astype(str).str.strip()
    data = data[(data["Text"] != "") & (data["Label"] != "")]

    if len(data) < 4 or data["Label"].nunique() < 2:
        raise ValueError("Add more labelled examples to the dataset before training.")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(data["Text"], data["Label"])
    return model, sorted(data["Label"].unique().tolist()), len(data)


model, categories, dataset_size = train_model()


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""
    error = None

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if not message:
            error = "Please enter a message first."
        elif len(message) > 2000:
            error = "Please keep the message under 2,000 characters."
        else:
            prediction = model.predict([message])[0]

    return render_template(
        "index.html",
        prediction=prediction,
        message=message,
        error=error,
        categories=categories,
        dataset_size=dataset_size
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # Render provides its own PORT; this also allows local testing.
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
