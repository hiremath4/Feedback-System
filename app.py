import sys
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify 
from src.preprocess import clean_text
from src.sentiment_analysis import analyze_sentiment

app = Flask(__name__)

@app.route('/feedback-system-oedg.onrender.com')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        user_input = request.form['user_input']
        cleaned_text = clean_text(user_input)
        sentiment = analyze_sentiment(cleaned_text)
        
        return jsonify({
            'input': user_input,
            'sentiment': sentiment
        })

if __name__ == '__main__':
    app.run(debug=True)


# Add 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.preprocess import preprocess_data
from src.analyze import analyze_sentiment

# Load and preprocess data
df = preprocess_data("data/data.csv")

# Check if DataFrame is empty
if df.empty:
    print("The data file is empty or not properly loaded.")
else:
    # Ensure the 'sentence' column exists
    if "sentence" in df.columns:
        # Apply sentiment analysis only to sentences longer than 5 characters
        df["sentiment"] = df["sentence"].apply(
            lambda x: analyze_sentiment(x) if len(str(x).strip()) > 5 else "Text too short"
        )

        # Save processed data
        df.to_csv("data.csv", index=False)
        print("Sentiment analysis complete. Check 'data.csv'.")
    else:
        print("The 'sentence' column is missing in the data.")
df["sentence"].apply(lambda x: analyze_sentiment(x) if len(str(x).strip()) > 5 else "Text too short")

