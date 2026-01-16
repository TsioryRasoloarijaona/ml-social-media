from fastapi import FastAPI , HTTPException
import numpy as np
import uvicorn
import utils
from types_classes import request_type
import pandas as pd
import joblib
from utils import utils 

try:
    model = joblib.load("rebuild/model/model.pkl")
    le_day = joblib.load("rebuild/encoder/le_day.pkl")
    le_type = joblib.load("rebuild/encoder/le_type.pkl")
except FileNotFoundError:
    print("Model or label encoder files not found. Please ensure they are in the correct path.")
    
app = FastAPI()



@app.post("/predict")
def predict(data : request_type):
    try:
        dt = pd.to_datetime(data.date_time)
        content_type = le_type.transform([data.content_type])[0]
        content_length = data.content_length
        follower_count = data.follower_count
        day = le_day.transform([dt.day_name()])[0]
        hour = dt.hour
        description_score = utils.calculate_description_score(data.description)
        hashtags_count = utils.count_hashtags(data.hashtags)
        description_word_count = utils.count_description_words(data.description)
        
        feature = np.array([[content_type,
                            content_length,
                            follower_count,
                            day,
                            hour,
                            description_score,
                            hashtags_count,
                            description_word_count]])
        prediction = model.predict(feature)[0]
        
        
        return {
            "predicted_engagement_rate": round(float(prediction), 2) ,
            "unit": "percentage",
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)