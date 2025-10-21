from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
from data_preprocessing import preprocess_data

def train_model():
    df = preprocess_data('./data/cleaned_tickets.csv')
    
    X = df['complaint_text']
    y = df['category']

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Save model and vectorizer
    joblib.dump(model, './model/ticket_classifier.pkl')
    joblib.dump(vectorizer, './model/tfidf_vectorizer.pkl')

    print("\n💾 Model and vectorizer saved successfully!")

if __name__ == "__main__":
    train_model()
