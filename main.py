from fastapi import FastAPI , HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

try:
    model = joblib.load("xgboost_instagram_model.pkl")
    le_type = joblib.load("le_content_type.pkl")
    le_lang = joblib.load("le_language.pkl")
except FileNotFoundError:
    print("Model or label encoder files not found. Please ensure they are in the correct path.")
    
app = FastAPI()

class PostRequest(BaseModel):
    type: str
    lang: str
    content_length: int
    description: str
    hashtags: str
    follower_count: int
    time: int

@app.post("/predict")
def predict(data : PostRequest):
    try:
        content_type = le_type.transform([data.type])[0]
        language = le_lang.transform([data.lang])[0]
        content_description = len(data.description)
        hashtags_count = data.hashtags.count("#") 
        features = np.array([[content_type,
                              language,
                              data.content_length,
                              content_description,
                              hashtags_count,
                              data.follower_count,
                              data.time]])
        log_prediction = model.predict(features)
        log_pred = log_prediction.item()
        prediction = np.exp(log_pred) - 1
        
        return {
            
            "predicted_engagement_rate": round(float(prediction), 2) ,
            "unit": "percentage",
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)