import streamlit as st
import joblib
import re
import pandas as pd
import plotly.express as px
import numpy as np
import os

# ------------------- STREAMLIT PAGE CONFIG -------------------
st.set_page_config(page_title="AI Ticket Classifier", page_icon="🤖", layout="wide")

# ------------------- CUSTOM CSS -------------------
st.markdown("""
    <style>
    body {
        background-color: #f9fafc;
        color: #111;
    }
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        color: #2b2d42;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        margin-bottom: 30px;
    }
    .metric-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


import os, requests, joblib

MODEL_PATH = "./model/best_model_day15.pkl"
MODEL_URL = "https://raw.githubusercontent.com/Arjun-murmu/AI_Ticket_Classifier/main/model/best_model_day15.pkl"  # your hosted file link

if not os.path.exists(MODEL_PATH):
    # download once
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
model = joblib.load(MODEL_PATH)


# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('./model/best_model_day15.pkl')
    vectorizer = joblib.load('./model/vectorizer_day15.pkl')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ------------------- LOAD DATA -------------------
@st.cache_data
def load_data():
    return pd.read_csv('./data/cleaned_tickets.csv')

df = load_data()

# ------------------- CATEGORY MAP -------------------
category_map = {
    0: "Login Issues",
    1: "Payment Problems",
    2: "Account Suspension",
    3: "Service Request",
    4: "Technical Error",
}

# Convert numeric to readable if needed
if df['category'].dtype in ['int64', 'float64']:
    df['category'] = df['category'].map(category_map)

# ------------------- CLEAN TEXT -------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ------------------- SIDEBAR NAV -------------------
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dashboard", "📈 Evaluate Model", "ℹ️ About"])

# ------------------- HOME PAGE -------------------
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>🎯 AI Customer Support Ticket Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Automate customer complaint categorization using Machine Learning 🤖</p>", unsafe_allow_html=True)

    user_input = st.text_area("🗣️ Enter your complaint below:", "Example: 'I can’t login to my account'", height=120)

    if st.button("🔍 Predict Category"):
        if user_input.strip():
            input_tfidf = vectorizer.transform([clean_text(user_input)])
            prediction = model.predict(input_tfidf)[0]

            # Convert numeric label to name
            if isinstance(prediction, (int, np.integer)):
                category_name = category_map.get(prediction, str(prediction))
            else:
                category_name = str(prediction)

            st.success(f"**Predicted Category:** {category_name}")

            # Try to show probability if model supports it
            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba(input_tfidf)) * 100
                st.info(f"📊 Model Confidence: {confidence:.2f}%")
            else:
                st.info("⚠️ This model does not support probability-based confidence display.")
        else:
            st.warning("⚠️ Please enter a complaint first.")

    st.markdown("---")
    st.caption("Developed by **Arjun Murmu** | Powered by Streamlit & Machine Learning 🤖")

# ------------------- DASHBOARD PAGE -------------------
elif page == "📊 Dashboard":
    st.markdown("<h1 class='main-title'>📈 Ticket Data Insights Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Interactive visual analytics for support ticket trends</p>", unsafe_allow_html=True)

    total_tickets = len(df)
    num_categories = df['category'].nunique()
    top_category = df['category'].value_counts().idxmax()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>📄 Total Tickets</h4><h2>{total_tickets}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>🏷️ Categories</h4><h2>{num_categories}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>🔥 Top Category</h4><h2>{top_category}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")

    col4, col5 = st.columns(2)

    with col4:
        st.write("### 📊 Category Distribution")
        category_counts = df['category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        fig = px.bar(
            category_counts,
            x='category',
            y='count',
            color='category',
            text='count',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.write("### 🥧 Category Percentage")
        fig2 = px.pie(
            df,
            names='category',
            title='Ticket Categories %',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig2, use_container_width=True)

# ------------------- EVALUATE MODEL PAGE -------------------
elif page == "📈 Evaluate Model":
    st.header("📈 Model Evaluation")
    st.write("Running evaluation on cleaned_tickets.csv ...")

    if os.path.exists("./model/model_evaluation_report.csv"):
        df_report = pd.read_csv("./model/model_evaluation_report.csv")
        st.table(df_report.T)
    else:
        st.info("Please run `python evaluation.py` in terminal to generate evaluation report, then refresh this page.")

# ------------------- ABOUT PAGE -------------------
elif page == "ℹ️ About":
    st.markdown("<h1 class='main-title'>ℹ️ About This Project</h1>", unsafe_allow_html=True)
    st.markdown("""
    **Project Title:** AI-Powered Customer Support Ticket Classifier  
    **Developer:** Arjun Murmu  
    **Tech Stack:** Python, Scikit-learn, NLTK, Streamlit, Plotly  
    **Goal:** Automate customer complaint categorization and reduce manual ticket sorting.

    **Model Pipeline:**  
    - 🧹 Text Preprocessing  
    - ✍️ TF-IDF Feature Extraction  
    - 🤖 Logistic Regression / SVM / XGBoost  
    - 📈 Accuracy Evaluation  

    ---
    📅 **Development Duration:** 20 Days  
    📍 **Status:** ✅ Completed & Functional  
    💡 **Next Goal:** Continuous Improvement and Deployment
    """)

    st.success("Project ready for demo presentation 🚀")
