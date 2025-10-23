import pandas as pd
import numpy as np
import joblib
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------- SETUP -------------------
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('./data/cleaned_tickets.csv')
df['category'] = df['category'].str.strip()

print("\n✅ Cleaned Categories:")
print(df['category'].unique())
print("✅ Data loaded successfully!")
print(df.head())

# ------------------- TEXT CLEANING -------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

df['complaint_text'] = df['text'].apply(clean_text)

# ------------------- LABEL ENCODING -------------------
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

X = df['complaint_text']
y = df['category_encoded']

# ------------------- FEATURE EXTRACTION -------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# ------------------- CHECK CLASS COUNTS -------------------
print("\n📊 Category distribution:")
print(pd.Series(label_encoder.inverse_transform(y)).value_counts())

min_class_count = pd.Series(y).value_counts().min()
if min_class_count < 2:
    print(f"\n⚠️ Warning: Some categories have only {min_class_count} sample(s). Stratified split disabled.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------- MODELS -------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

results = []

# ------------------- TRAIN AND EVALUATE -------------------
for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ {name} Accuracy: {acc:.4f}")
    results.append((name, acc))

    # Decode labels back for readable report
    print("\n📋 Classification Report:")
    print(classification_report(label_encoder.inverse_transform(y_test),
                                label_encoder.inverse_transform(y_pred)))

# ------------------- COMPARE RESULTS -------------------
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
print("\n📊 Model Comparison:\n", results_df)

# ------------------- PLOT ACCURACY COMPARISON -------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x='Model', y='Accuracy', palette='coolwarm')
plt.title("Model Comparison - Accuracy (Day 15)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------- CONFUSION MATRIX (Best Model) -------------------
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
labels = label_encoder.classes_

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ------------------- SAVE BEST MODEL -------------------
joblib.dump(best_model, f'./model/best_model_day15.pkl')
joblib.dump(vectorizer, f'./model/vectorizer_day15.pkl')
joblib.dump(label_encoder, f'./model/label_encoder_day15.pkl')

print(f"\n🏆 Best Model: {best_model_name} saved successfully!")
