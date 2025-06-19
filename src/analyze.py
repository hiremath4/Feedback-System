from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Initialize Watson NLU
API_KEY = "CL6Z5EKXRMYt8dV5tdpikjED1pTwMJl8putP3sFaCn6z"  
SERVICE_URL = "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/7eefac19-8970-4eea-b52c-a6d48009e84d"


authenticator = IAMAuthenticator(API_KEY)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(SERVICE_URL)

def analyze_sentiment(text):
    if len(text.strip()) < 5:
        raise ValueError("Input text is too short for analysis.")

    try:
        response = nlu.analyze(
            text=text,
            features=Features(sentiment=SentimentOptions()),
            language='en'  # Specify language if known
        ).get_result()

        sentiment_label = response['sentiment']['document']['label']
        sentiment_score = response['sentiment']['document']['score']

        return {
            'text': text,
            'sentiment': sentiment_label,
            'score': sentiment_score
        }

    except Exception as e:
        print(f"Watson API Error: {e}")
        return None
