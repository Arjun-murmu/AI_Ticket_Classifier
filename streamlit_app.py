import streamlit as st
import joblib
import re
import pandas as pd
import plotly.express as px
import numpy as np
import os
import evaluation

# ------------------- STREAMLIT PAGE CONFIG -------------------
st.set_page_config(page_title="AI Ticket Classifier", page_icon="🤖", layout="wide")

# Custom CSS for better look
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

# ------------------- CACHE MODEL -------------------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('./model/ticket_classifier_optimized.pkl')
    vectorizer = joblib.load('./model/tfidf_vectorizer_optimized.pkl')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ------------------- LOAD DATA -------------------
@st.cache_data
def load_data():
    return pd.read_csv('./data/cleaned_tickets.csv')

df = load_data()

# ------------------- TEXT CLEAN FUNCTION -------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ------------------- SIDEBAR -------------------
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dashboard", "ℹ️ About", "📈 Evaluate Model"])

# ------------------- HOME PAGE -------------------
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>🎯 AI Customer Support Ticket Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Automate customer complaint categorization using Machine Learning 🤖</p>", unsafe_allow_html=True)

    user_input = st.text_area("🗣️ Enter your complaint below:", "Example: 'I can’t login to my account'", height=120)

    if st.button("🔍 Predict Category"):
        if user_input.strip():
            input_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(input_tfidf)[0]
            confidence = np.max(model.predict_proba(input_tfidf)) * 100

            st.success(f"**Predicted Category:** {prediction}")
            st.info(f"📊 Model Confidence: {confidence:.2f}%")
        else:
            st.warning("⚠️ Please enter a complaint first.")

    st.markdown("---")
    st.caption("Developed by **Arjun Murmu** | Powered by Streamlit & Machine Learning 🤖")

# ------------------- DASHBOARD PAGE -------------------
elif page == "📊 Dashboard":
    st.markdown("<h1 class='main-title'>📈 Ticket Data Insights Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Interactive visual analytics for support ticket trends</p>", unsafe_allow_html=True)

    # Summary metrics
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

    # Charts
    col4, col5 = st.columns(2)

    with col4:
        st.write("### 📊 Category Distribution")
        category_counts = df['category'].value_counts().reset_index(name='count')
        category_counts = category_counts.rename(columns={'index': 'category'})
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

# ------------------- Evaluate Model-------------------
elif page == "📈 Evaluate Model":
    st.header("📈 Model Evaluation")
    st.write("Running evaluation on cleaned_tickets.csv ...")
    # We will reuse the evaluate function but adapt to Streamlit: render Plotly figure inside Streamlit.
    # Minimal approach: run evaluation.py's logic here (or import a function from evaluation.py that returns metrics+fig).
    # For now you can run the script externally and show the saved CSV:
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
    - 🤖 Logistic Regression Model  
    - 📈 Accuracy Evaluation  

    ---
    📅 **Development Duration:** 20 Days  
    📍 **Status:** ✅ Completed & Functional  
    💡 **Next Goal:** Model Optimization & Deployment (Day 13)
    """)

    st.success("Project ready for demo presentation 🚀")
