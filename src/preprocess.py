import pandas as pd
import re

def clean_text(text):
    """Basic text cleaning function."""
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    return text.lower().strip()

def preprocess_data(file_path):
    """Loads and preprocesses data."""
    
    df = pd.read_csv(file_path)
    df['clean_text'] = df['sentence'].apply(clean_text)
    return df

def new_func(file_path):
    return file_path

if __name__ == "__main__":
    df = preprocess_data("data/data.csv")
    print(df.head())
