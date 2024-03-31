from typing import Dict
from pydantic import BaseModel

class PredictModel(BaseModel):
    model_name: str
    parameters: str
    api_key:str