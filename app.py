from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("model.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    msg = data["message"]

    # Keyword-based instant spam detection (Indian scam words)
   
    fraud_keywords = [
        "aadhaar", "aadhar", "kyc", "loan approved", "loan", "account blocked",
        "pan card", "verify your account", "update your account", 
        "your bank account", "blocked", "free gift", "click the link",
        "urgent update", "offer", "prize", "winner","otp","" "password",
        "credit card", "debit card", "bank details", "net banking","bank",
        "immediate action", "limited time", "risk-free", "act now","exclusive deal",
        "call now", "don't miss", "congratulations", "claim your reward","account suspended""account verification",
        "suspicious activity", "security alert", "important notice", "final notice", "payment required", "sensitive information",
        "verify identity", "identity theft", "financial information", "personal information", "click here", "visit this link", "download attachment",
        "urgent response", "confidential information", "account update", "service interruption", "unauthorized access", "account locked",
        "suspicious login", "unusual activity", "link", "customer support", "help desk", "technical support", "fraudulent activity",
        "scam", "phishing", "malware", "virus", "spyware", "ransomware", "hacked", "data breach", "security breach"]

    for word in fraud_keywords:
        if word.lower() in msg.lower():
            return jsonify({
                "label": "spam",
                "confidence": 0.99
            })

    # Machine Learning prediction (your model)

    proba = model.predict_proba([msg])[0][1]
    pred = model.predict([msg])[0]

    return jsonify({
        "label": "spam" if pred == 1 else "ham",
        "confidence": float(proba)
    })

if __name__ == "__main__":
    app.run(debug=True)
    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)