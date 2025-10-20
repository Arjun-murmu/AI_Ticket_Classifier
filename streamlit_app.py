import streamlit as st
import joblib
import re
import pandas as pd
import plotly.express as px
import numpy as np

# Load model and vectorizer
model = joblib.load('./model/ticket_classifier.pkl')
vectorizer = joblib.load('./model/tfidf_vectorizer.pkl')

#load dataset for visualization
df = pd.read_csv('./data/cleaned_tickets.csv')

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Streamlit UI
st.set_page_config(page_title="AI Ticket Classifier", page_icon="🤖", layout="wide")

# Sidebar Navigation
st.sidebar.title("🎯 AI Customer Support Ticket Classifier")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dashboard", "ℹ️ About"])

# ------------------ HOME PAGE ------------------
if page == "🏠 Home":
    st.title("🎯 AI Customer Support Ticket Classifier")
    st.markdown("""
    This app automatically classifies customer complaints into categories using **Machine Learning** 🤖.  
    Try it below ⬇️
    """)

    # Input box
    st.write("Type your complaint below to get the predicted category:")
    st.markdown("""
    Welcome to the **AI Ticket Classifier App** 🤖  
    This system automatically categorizes customer complaints into categories like:
    - 💳 **Billing Issue**  
    - 🔐 **Account Issue**  
    - 🌐 **Network Issue**  
    - ⚙️ **Technical Issue**
    """)
    # Input box
    st.subheader("🗣️ Enter your complaint:")
    user_input = st.text_area("Example: 'I can’t login to my account'", height=120)
    # user_input = st.text_area("📝 Enter your complaint:", height=120)

    if st.button("🔍 Predict Category"):
        if user_input.strip():
            input_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(input_tfidf)[0]
            # Confidence score
            confidence = np.max(model.predict_proba(input_tfidf)) * 100

            st.success(f"**Predicted Category:** {prediction}")
            st.info(f"📊 Model Confidence: {confidence:.2f}%")
        else:
            st.warning("⚠️ Please enter a complaint first.")

    st.markdown("---")
    st.caption("Developed by **Arjun Murmu** | Powered by Machine Learning 🤖")

# ------------------ DASHBOARD PAGE ------------------
elif page == "📊 Dashboard":
    st.title("📈 Ticket Data Insights Dashboard")
    st.markdown("Visualize your dataset and see the category distribution:")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Category Distribution")

        # Fix: Correct column names after value_counts()
        category_counts = df['category'].value_counts().reset_index(name='count')
        category_counts = category_counts.rename(columns={'index': 'category'})

        fig = px.bar(
            category_counts,
            x='category',
            y='count',
            color='category',
            text='count',
            labels={'category': 'Category', 'count': 'Count'},
            title='Ticket Category Counts'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("### Category Percentage")
        fig2 = px.pie(df, names='category', title='Ticket Categories %', color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig2, use_container_width=True)

# ------------------ ABOUT PAGE ------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    **Project Title:** AI-Powered Customer Support Ticket Classifier  
    **Developer:** Arjun Murmu  
    **Tech Stack:** Python, Scikit-learn, NLTK, Streamlit, Plotly  
    **Goal:** To automate customer complaint categorization and reduce manual ticket sorting.  

    **Model Details:**  
    - Text Preprocessing: Tokenization, Stopword Removal  
    - Feature Extraction: TF-IDF Vectorization  
    - Classifier: Logistic Regression  
    - Evaluation: Accuracy, Confusion Matrix, Precision & Recall  

    ---
    📅 **Development Duration:** 20 Days  
    📍 **Status:** ✅ Completed & Functional
    """)

    st.success("Project fully functional and ready for submission 🚀")      


