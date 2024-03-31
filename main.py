from typing import Any
from fastapi import FastAPI, HTTPException, Body, Header, Query
from model_train_algorithms.RandomForest import train_and_store_random_forest
from model_train_algorithms.SVM import train_and_store_SVM
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.fetch_shap_data import fetch_shap_force_plot
from utils.fetch_user_models import fetch_user_models
from utils.fetch_model_details import fetch_model_details
from utils.fetch_shap_data import fetch_shap_summary_plot
from models.predict_model import PredictModel
from model_predict.model_predict import model_predict
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


@app.post("/predict")
def predict_for_model(req: PredictModel):
    if len(req.model_name) == 0 or len(req.parameters) == 0 or len(req.api_key) == 0:
        return JSONResponse(content={"message": "You have not provided all the parameters!"}, status_code=400)  
    model_params={}
    try:
        model_params = json.loads(req.parameters)
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", str(e))
        return JSONResponse(content={"message": "You have not provided a valid parameters object!"}, status_code=400)  
    key_to_check = "param_array"
    if key_to_check not in model_params:
        return JSONResponse(content={"message": "The parameters should have an object that has a key called param_array and that array should contain the parameters!"}, status_code=400)    
    prediction = model_predict(req.model_name, model_params)
    response = {
        "prediction": prediction
    }
    return JSONResponse(content={"message": response}, status_code=200)

@app.post("/api/models/train")
async def root(   data: Any = Body(...),
    model_name: Any = Body(...),
    model_params: Any = Body(...),
    email: Any = Body(...)
    ):

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Invalid parameters. Please provide valid values.")
    if model_name == "Random Forest":
        resp = train_and_store_random_forest(data, model_params,email)
    elif model_name == "SVM":
        resp = train_and_store_SVM(data, model_params,email)

    resp = json.dumps({
        "model_name": str(resp)
    })
    
    return JSONResponse(content={"message": resp}, status_code=200)

@app.get("/api/models")
async def get_models_for_user(email: str = Query(..., description="The email address of the user to retrieve.")):
    
    if len(email) != 0:
        res = fetch_user_models(email)
        return res
    else:
        return JSONResponse(content={"message": "There was a problem processing request"}, status_code=400)


@app.get("/api/force_plot")
async def get_shap_force_plot(model_name: str = Query(..., description="The email address of the user to retrieve.")):
    
    if len(model_name) != 0:
        res =  fetch_shap_force_plot(model_name)
        return JSONResponse(content={"data":res}, status_code=200)
    else:
        return JSONResponse(content={"message": "There was a problem processing request"}, status_code=400)

@app.get("/api/summary_plot")
async def get_shap_summary_plot(model_name: str = Query(..., description="The email address of the user to retrieve.")):
    
    
    if len(model_name) != 0:
        res =  fetch_shap_summary_plot(model_name)
        return JSONResponse(content={"data": res}, status_code=200)
        
    else:
        return JSONResponse(content={"message": "There was a problem processing request"}, status_code=400)



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8086, log_level="info")

