from ibm_watson import NaturalLanguageUnderstandingV1  # type: ignore
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator  # type: ignore
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions  # type: ignore
import pandas as pd
WATSON_API_KEY = "CL6Z5EKXRMYt8dV5tdpikjED1pTwMJl8putP3sFaCn6z"
WATSON_URL = "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/7eefac19-8970-4eea-b52c-a6d48009e84d"


# Initialize Watson API
authenticator = IAMAuthenticator(WATSON_API_KEY)
nlu = NaturalLanguageUnderstandingV1(version="2022-04-07", authenticator=authenticator)
nlu.set_service_url(WATSON_URL)

def analyze_sentiment(text):
    """Calls Watson API to analyze sentiment."""
    try:
        response = nlu.analyze(text=text, features=Features(sentiment=SentimentOptions())).get_result()
        return response["sentiment"]["document"]["label"]
    except Exception as e:
        print("Error:", e)
        return "neutral"

if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    df["sentiment"] = df["sentence"].apply(analyze_sentiment)
    df.to_csv("data/data.csv", index=False)
    print(df.head())
