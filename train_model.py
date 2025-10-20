from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

def train_model(data_path):
    from data_preprocessing import preprocess_data
    df = preprocess_data(data_path)

    X = df['complaint_text']
    y = df['category']

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "./models/ticket_classifier.pkl")
    joblib.dump(vectorizer, "./models/vectorizer.pkl")

    print("✅ Model training complete and saved!")
