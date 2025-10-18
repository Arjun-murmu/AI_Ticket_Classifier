import streamlit as st
import joblib
import re
import numpy as np

# Load model and vectorizer
model = joblib.load('./model/ticket_classifier.pkl')
vectorizer = joblib.load('./model/tfidf_vectorizer.pkl')

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Streamlit UI
st.set_page_config(page_title="AI Ticket Classifier", page_icon="🎯", layout="centered")
st.title("🎯 AI Customer Support Ticket Classifier")
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
