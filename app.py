import pickle

# Load the saved model and vectorizer
with open("D:/AI_Ticket_Classifier/AI_Ticket_Classifier/model/ticket_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("D:/AI_Ticket_Classifier/AI_Ticket_Classifier/model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Run a prediction loop
print("🎯 AI Customer Support Ticket Classifier 🎯")
print("Type 'exit' to stop.\n")

while True:
    text = input("Enter your complaint: ")

    if text.lower() == "exit":
        print("Goodbye! 👋")
        break

    # Convert text to vector
    text_tfidf = vectorizer.transform([text])

    # Predict category
    prediction = model.predict(text_tfidf)[0]
    print(f"Predicted Category: {prediction}\n")
