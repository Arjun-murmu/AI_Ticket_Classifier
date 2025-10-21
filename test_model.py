import joblib

def test_prediction():
    model = joblib.load('./model/ticket_classifier.pkl')
    vectorizer = joblib.load('./model/tfidf_vectorizer.pkl')

    samples = [
        "Internet not working since morning",
        "Payment issue with my recent order",
        "Unable to login to my account",
        "Please update my address details"
    ]

    for text in samples:
        X_vec = vectorizer.transform([text])
        pred = model.predict(X_vec)[0]
        print(f"Text: {text}\nPredicted Category: {pred}\n")

if __name__ == "__main__":
    test_prediction()
