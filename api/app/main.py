from fastapi import FastAPI

from iris import IrisModel
from inference import predict_iris

app = FastAPI()

@app.post("/predict/")
async def predict(model: IrisModel):
    iris_type = predict_iris(model)
    return {
        "prediction": iris_type
    }
