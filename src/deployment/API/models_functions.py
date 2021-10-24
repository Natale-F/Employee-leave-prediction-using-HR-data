import joblib
import os
import numpy as np
import pandas as pd


def load_model():
    """
    Load the model from the disk.

    Returns:
         (xgboost.XGBClassifier): best trained xgboost model
    """
    model_path = os.path.join('pipeline', 'machine_learning', 'optimized_xgboost.pkl')
    model = joblib.load(model_path)
    print('Model loaded from disk')
    return model


def transform_input(received_data: list):
    """
    Transform received input data in a dataframe.

    Args:
        received_data (List[InputData]): a list of InputData objects
                                         --> see: src.deployment.API.data_structure

    Returns:
        reshaped_data (pd.DataFrame): data reshaped in a dataframe
    """
    reshaped_data = []
    for input_data in received_data:
        dico_data = {
            'Education': input_data.education,
            'JoiningYear': input_data.joining_year,
            'City': input_data.city,
            'PaymentTier': input_data.payment_tier,
            'Age': input_data.age,
            'Gender': input_data.gender,
            'EverBenched': input_data.ever_benched,
            'ExperienceInCurrentDomain': input_data.experience_in_current_domain
        }
        reshaped_data.append(dico_data)
    # transform into a dataframe
    reshaped_data = pd.DataFrame(reshaped_data)
    return reshaped_data


def transform_prediction(array_prediction: np.array):
    """
    Transform predicted values into readable format.

    Args:
        array_prediction (np.array): predicted class (0 or 1)

    Returns:
        list_prediction (list): predicted class in english
    """
    pred_series = pd.Series(array_prediction)
    pred_series[pred_series == 0] = 'Stay'
    pred_series[pred_series == 1] = 'Leave'
    list_prediction = pred_series.to_list()
    list_prediction_for_api = [{'leave_or_not': x} for x in list_prediction]
    return list_prediction_for_api