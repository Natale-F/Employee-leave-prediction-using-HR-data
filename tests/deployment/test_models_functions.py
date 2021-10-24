import os
import pandas as pd
import joblib
from src.deployment.API.models_functions import load_model, transform_input, transform_prediction
from src.preprocessing.pipeline_preprocessing import prepare_raw_data
from src.deployment.API.data_structure import InputData

path_pipeline = os.path.join('pipeline', 'preprocessing', 'preprocessing_model.pkl')
preprocessing_pipeline = joblib.load(path_pipeline)

# load data
train_data_path = os.path.join('data', 'training', 'x_train.csv')
X_train = pd.read_csv(train_data_path)
usecols = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender',
           'EverBenched', 'ExperienceInCurrentDomain']
X_train = X_train[usecols]
X_prepared = prepare_raw_data(X_train, preprocessing_pipeline)

# load the model
model = load_model()


def test_load_model():
    """
    Test if the loaded model is the one you want to have.
    """
    assert model != None
    assert model.predict(X_prepared) is not None


def test_transform_input():
    """
    Test if transform_input function do the transformation that we want (List[InputData] --> pd.DataFrame).
    """
    dico_test = {
        'Education': 'Bachelors',
        'JoiningYear': 2018,
        'City': "Bangalore",
        'PaymentTier': 1,
        'Age': 25,
        'Gender': 'Male',
        'EverBenched': 'Yes',
        'ExperienceInCurrentDomain': 5
    }
    test_InputData = [
        InputData(education=dico_test['Education'], joining_year=dico_test['JoiningYear'],
                  city=dico_test['City'], payment_tier=dico_test['PaymentTier'],
                  age=dico_test['Age'], gender=dico_test['Gender'],
                  ever_benched=dico_test['EverBenched'],
                  experience_in_current_domain=dico_test['ExperienceInCurrentDomain'])
                    ]
    df_test = transform_input(test_InputData)
    assert type(df_test) == pd.DataFrame
    for col in df_test.columns:
        assert df_test[col][0] == dico_test[col]

def test_transform_prediction():
    """
    Tests whether the function transforms the predicted variables into categorical values.
    """
    y_pred = model.predict(X_prepared[:100])
    list_set_pred = list(set(y_pred))
    # transform the data
    y_pred_transformed = transform_prediction(y_pred)

    # test whether the predicted values are those that should be
    assert len(list_set_pred) == 2
    assert 0 in list_set_pred
    assert 1 in list_set_pred
    # test if returned prediction are in a good shape
    for pred in y_pred_transformed:
        assert 'leave_or_not' in pred.keys()
        assert 'Stay' or 'Leave' in pred.values()
