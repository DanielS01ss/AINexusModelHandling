from typing import Any
from fastapi import FastAPI, HTTPException, Body, Header, Query
from model_train_algorithms.RandomForest import train_and_store_random_forest
from model_train_algorithms.SVM import train_and_store_SVM
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.fetch_user_models import fetch_user_models
from utils.fetch_model_details import fetch_model_details
import sys
import uvicorn
import json
import jwt

app = FastAPI()

origin_paterns = '*'


@app.get("/api/model/details")
def model_details(model_name: str = Query(..., description="The email address of the user to retrieve.")):
    if len(model_name) != 0:
        result = fetch_model_details(model_name)    
        print("result is:")
        print(result)
        if result['code'] == 200:
            return {"content": result, "code":200, "msg":"Success"}
        elif result['code'] == 404:
            return {"content": [], "code":404, "msg": result['msg']}
        else :
            return {"content": [], "code":500, "msg": result['msg']}
    else:
        raise HTTPException(status_code=404, detail="Item not found")


@app.post("/api/models/train")
async def root(   data: Any = Body(...),
    model_name: Any = Body(...),
    model_params: Any = Body(...),
    email: Any = Body(...)
    ):

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Invalid parameters. Please provide valid values.")
    if model_name == "Random Forest":
        train_and_store_random_forest(data, model_params,email)
    elif model_name == "SVM":
        train_and_store_SVM(data, model_params,email)

    return JSONResponse(content={"message": "Request processed successfully"}, status_code=200)

@app.get("/api/models")
async def get_models_for_user(email: str = Query(..., description="The email address of the user to retrieve.")):
    
    if len(email) != 0:
        res = fetch_user_models(email)
        return res
    else:
        return JSONResponse(content={"message": "There was a problem processing request"}, status_code=400)

@app.get("/python-info")
def python_info():
    return {"Python Executable": sys.executable, "Python Version": sys.version}

