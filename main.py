from pyexpat import features

from fastapi import FastAPI , HTTPException
import uvicorn
from types_classes import request_type
import joblib
from utils.mapper_request import transform_request_to_features

try:
    model = joblib.load("rebuild/model/model_v7.pkl")
except FileNotFoundError:
    print("Model or label encoder files not found. Please ensure they are in the correct path.")
    
app = FastAPI()



@app.post("/predict")
def predict(data : request_type):
    try:
        feature = transform_request_to_features(data)
        prediction = model.predict(feature)[0] * 10
        risk_level = "Low" if prediction < 1 else "Medium" if prediction < 3 else "High"
        return {
            "predicted_engagement_rate": round(float(prediction), 2) ,
            "unit": "percentage",
            "risk_level": risk_level
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)