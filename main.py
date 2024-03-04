from typing import Any
from fastapi import FastAPI, HTTPException, Body
from model_train_algorithms.RandomForest import train_and_store_random_forest
from model_train_algorithms.SVM import train_and_store_SVM
from fastapi.responses import JSONResponse
import json

app = FastAPI()


@app.post("/api/models/train")
async def root(   data: Any = Body(...),
    model_name: Any = Body(...),
    model_params: Any = Body(...) ):

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Invalid parameters. Please provide valid values.")
    if model_name == "Random Forest":
        train_and_store_random_forest(data, model_params)
    elif model_name == "SVM":
        train_and_store_SVM(data, model_params)

    return JSONResponse(content={"message": "Request processed successfully"}, status_code=200)
