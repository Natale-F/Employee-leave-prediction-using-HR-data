import os
import joblib
import webbrowser
from fastapi import FastAPI
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from src.deployment.API.data_structure import InputData, Prediction
from src.deployment.API.models_functions import load_model, transform_input, transform_prediction
from src.preprocessing.pipeline_preprocessing import prepare_raw_data


# meta groups
tags_metadata = [{"name": "Prediction"}]

# app initialisation
app = FastAPI(title='Employee Leave Machine Learning API',
              description="This API allows you to request a Machine Learning model to know if an employee"
                          " want to leave the company!",
              version="1.0.0",
              openapi_tags=tags_metadata)

# allows request from all ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load preprocessing pipeline
path_pipeline = os.path.join('pipeline', 'preprocessing', 'preprocessing_model.pkl')
preprocessing_pipeline = joblib.load(path_pipeline)

# load the model
model = load_model()

# opens the documentation when the API is launched
webbrowser.open('http://127.0.0.1:8000/docs')


@app.post('/employee',
          response_model=List[Prediction],
          tags=['Prediction'])
def post_request_for_prediction(received_data: List[InputData]):
    """
    Receives data, prepare them and returns the prediction.
    """
    # transform received data into a dataframe
    df_received_data = transform_input(received_data)
    # do the preprocessing on the data
    prepared_for_model_data = prepare_raw_data(df_received_data,
                                               preprocessing_pipeline)
    # predict the class
    predicted_output = model.predict(prepared_for_model_data)
    # reshape the output
    returned_prediction = transform_prediction(predicted_output)
    return returned_prediction


