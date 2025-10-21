import pandas as pd
import re

def clean_text(text):
    """Clean and normalize complaint text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(file_path='./data/cleaned_tickets.csv'):
    """Load and clean the dataset."""
    df = pd.read_csv(file_path)
    
    # Remove duplicates and missing values
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['complaint_text', 'category'], inplace=True)
    
    # Clean text
    df['complaint_text'] = df['complaint_text'].apply(clean_text)
    
    print("✅ Data cleaned successfully!")
    print(df.head())
    return df

if __name__ == "__main__":
    preprocess_data()
