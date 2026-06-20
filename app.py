# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import re

app = Flask(__name__)

# ---------------- Load Models and TF-IDF ----------------
tfidf = pickle.load(open("tfidf.pkl", "rb"))
nb_model = pickle.load(open("nb_model.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))
dt_model = pickle.load(open("dt_model.pkl", "rb"))

# Dictionary to map model names
models = {
    "nb": nb_model,
    "lr": lr_model,
    "dt": dt_model
}

# ---------------- Helper function ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("email", "")
    model_name = data.get("model", "nb")

    # Clean the text
    cleaned = clean_text(email_text)

    # Convert to TF-IDF vector
    vect = tfidf.transform([cleaned])

    # Select model
    model = models.get(model_name, nb_model)

    # Prediction
    pred = model.predict(vect)[0]

    # Probability (confidence)
    prob = model.predict_proba(vect)[0]
    confidence = round(max(prob) * 100, 2)

    result = "SPAM" if pred == 1 else "NOT SPAM"

    # Return JSON response
    return jsonify({
        "prediction": result,
        "confidence": confidence,
        "model": model_name
    })

# ---------------- Run the app ----------------
if __name__ == "__main__":
    print("Flask app is starting on http://127.0.0.1:5000/")
    app.run(debug=True)