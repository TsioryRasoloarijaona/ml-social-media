from textblob import TextBlob

def calculate_description_score(description):
    if not description:
        return 0.0
    blob = TextBlob(str(description))
    return blob.sentiment.polarity

def count_description_words(description):
    if not description:
        return 0
    return len(str(description).split())

def count_hashtags(hashtags):
    if not hashtags:
        return 0
    return str(hashtags).count("#")