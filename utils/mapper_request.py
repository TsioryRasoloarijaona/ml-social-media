from . import date_day_hour
from . import description_and_hashtag
from datetime import datetime
import numpy as np

def transform_request_to_features(data) -> np.ndarray:
    dt = datetime.fromisoformat(data.date_time.replace("Z", ""))
    hour = dt.hour
    day_num = dt.weekday()
    day_sin, day_cos = date_day_hour.encode_day_cyclic(day_num)
    description_score = description_and_hashtag.description_performance_score(data.description)
    hashtag_quality_score = description_and_hashtag.hashtag_quality_score(data.description, data.hashtags)


    features = [
        data.follower_count,
        hour,
        description_score,
        hashtag_quality_score,
        day_sin,
        day_cos,
    ]

    return np.array([features])