# evaluation.py
import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# --------- CONFIG: change these if your filenames differ ----------
MODEL_PATHS = [
    "./model/ticket_classifier_optimized.pkl",
    "./model/ticket_classifier.pkl"
]
VECT_PATHS = [
    "./model/tfidf_vectorizer_optimized.pkl",
    "./model/tfidf_vectorizer.pkl"
]
DATA_PATH = "./data/cleaned_tickets.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
# -----------------------------------------------------------------

def find_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    return None

def load_model_and_vectorizer():
    model_file = find_existing(MODEL_PATHS)
    vec_file = find_existing(VECT_PATHS)
    if model_file is None or vec_file is None:
        raise FileNotFoundError(f"Model or vectorizer not found. Searched:\nModels:{MODEL_PATHS}\nVecs:{VECT_PATHS}")
    model = joblib.load(model_file)
    vectorizer = joblib.load(vec_file)
    print(f"Loaded model: {model_file}")
    print(f"Loaded vectorizer: {vec_file}")
    return model, vectorizer

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # Accept different column names used earlier: text / clean_text / complaint_text
    if 'clean_text' in df.columns:
        text_col = 'clean_text'
    elif 'complaint_text' in df.columns:
        text_col = 'complaint_text'
    elif 'text' in df.columns:
        text_col = 'text'
    else:
        raise KeyError("Could not find text column in data. Expected one of: clean_text, complaint_text, text")
    if 'category' not in df.columns:
        raise KeyError("Data must contain a 'category' column")
    df = df[[text_col, 'category']].dropna().reset_index(drop=True)
    df = df.rename(columns={text_col: 'text'})
    return df

def evaluate():
    model, vectorizer = load_model_and_vectorizer()
    df = load_data()

    # If dataset is tiny or has classes with single sample, remove classes <2 to allow stratify
    counts = df['category'].value_counts()
    small_classes = counts[counts < 2].index.tolist()
    if small_classes:
        print("Warning: removing categories with <2 samples for evaluation:", small_classes)
        df = df[df['category'].isin(counts[counts >= 2].index)]

    X = df['text']
    y = df['category']

    # split (we split raw text then transform using the loaded vectorizer)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_test_tfidf = vectorizer.transform(X_test)

    y_pred = model.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {accuracy:.4f}, Precision (weighted): {precision:.4f}, Recall (weighted): {recall:.4f}, F1 (weighted): {f1:.4f}")

    # Confusion matrix (Plotly)
    labels = sorted(np.unique(np.concatenate((y_test, y_pred))))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig = px.imshow(cm_df,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=labels, y=labels,
                    color_continuous_scale='Blues',
                    title="Confusion Matrix")
    fig.update_layout(width=700, height=600)
    fig.show()

    # Save a small CSV report
    report_df = pd.DataFrame([{
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'num_test_samples': len(y_test)
    }])
    report_df.to_csv("./model/model_evaluation_report.csv", index=False)
    print("\nSaved model_evaluation_report.csv to ./model/")

if __name__ == "__main__":
    evaluate()
