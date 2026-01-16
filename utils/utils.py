from textblob import TextBlob
from datetime import datetime
import numpy as np

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



def extract_day_of_week(date_time: str) -> int:
    dt = datetime.fromisoformat(date_time.replace("Z", ""))
    return dt.weekday()


def encode_day_cyclic(day_num: int) -> tuple[float, float]:
    sin_day = np.sin(2 * np.pi * day_num / 7)
    cos_day = np.cos(2 * np.pi * day_num / 7)
    return sin_day, cos_day

def encode_content_type(content_type: str) -> dict:
    encoded = {
        "content_type_video": 0,
        "content_type_text": 0,
        "content_type_mixed": 0,
    }
    if content_type in ["video", "text", "mixed"]:
        encoded[f"content_type_{content_type}"] = 1
    return encoded

def transform_request_to_features(data) -> np.ndarray:
    dt = datetime.fromisoformat(data.date_time.replace("Z", ""))
    hour = dt.hour
    day_num = dt.weekday()
    day_sin, day_cos = encode_day_cyclic(day_num)
    description_score = calculate_description_score(data.description)
    hashtag_count = count_hashtags(data.hashtags)
    description_word_count = count_description_words(data.description)
    content_type_encoded = encode_content_type(data.content_type)

    features = [
        data.content_length,
        data.follower_count,
        hour,
        description_score,
        hashtag_count,
        description_word_count,
        day_sin,
        day_cos,
        content_type_encoded["content_type_mixed"],
        content_type_encoded["content_type_text"],
        content_type_encoded["content_type_video"],
    ]

    return np.array([features])


