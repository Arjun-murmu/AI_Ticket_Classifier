import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ------------------- DOWNLOAD NLTK RESOURCES -------------------
nltk.download('stopwords')
nltk.download('wordnet')

# ------------------- LOAD DATA -------------------
df = pd.read_csv('./data/cleaned_tickets.csv')
print("✅ Data loaded successfully!")
print(df.head())

# ------------------- CLEAN TEXT -------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

df['complaint_text'] = df['text'].apply(clean_text)

# ------------------- FEATURE EXTRACTION -------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['complaint_text'])
y = df['category']

# ------------------- FIX: Remove categories with fewer than 2 samples -------------------
category_counts = y.value_counts()
valid_classes = category_counts[category_counts >= 2].index
df = df[df['category'].isin(valid_classes)]

# Recreate X and y after filtering
X = vectorizer.fit_transform(df['complaint_text'])
y = df['category']

# ------------------- TRAIN-TEST SPLIT -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------- MODEL 1: Logistic Regression (GridSearch) -------------------
param_grid_lr = {'C': [0.1, 1, 10]}
grid_lr = GridSearchCV(LogisticRegression(max_iter=500), param_grid_lr, cv=3, n_jobs=-1)
grid_lr.fit(X_train, y_train)
lr_best = grid_lr.best_estimator_
y_pred_lr = lr_best.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# ------------------- MODEL 2: Linear SVM -------------------
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

# ------------------- MODEL 3: Random Forest -------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# ------------------- COMPARE RESULTS -------------------
print("\n📊 Model Comparison:")
print(f"Logistic Regression: {acc_lr:.4f}")
print(f"Linear SVC:          {acc_svm:.4f}")
print(f"Random Forest:       {acc_rf:.4f}")

# ------------------- BEST MODEL SELECTION -------------------
best_model, best_name, best_acc = max(
    [(lr_best, "Logistic Regression", acc_lr),
     (svm_model, "Linear SVC", acc_svm),
     (rf_model, "Random Forest", acc_rf)],
    key=lambda x: x[2]
)

print(f"\n🏆 Best Model: {best_name} with Accuracy: {best_acc:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, best_model.predict(X_test)))

# ------------------- SAVE MODEL & VECTORIZER -------------------
joblib.dump(best_model, './model/ticket_classifier_optimized.pkl')
joblib.dump(vectorizer, './model/tfidf_vectorizer_optimized.pkl')
print("\n💾 Saved optimized model successfully!")
