import os
import joblib
import pandas as pd
import numpy as np
import sklearn
from src.preprocessing.pipeline_preprocessing import prepare_raw_data
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# load the prepared pipeline
path_pipeline = os.path.join('pipeline', 'preprocessing', 'preprocessing_model.pkl')
path_pipeline_linear = os.path.join('pipeline', 'preprocessing', 'preprocessing_linear_model.pkl')

preprocessing_pipeline = joblib.load(path_pipeline)
preprocessing_pipeline_linear = joblib.load(path_pipeline_linear)


def test_one_hot_encoding():
    """
    Test that the one-hot transformer is well stored (the code is based on a storage order).
    """
    one_hot_transformer = preprocessing_pipeline.transformers_[0]
    one_hot_transformer_lin = preprocessing_pipeline_linear.transformers_[0]

    assert one_hot_transformer[0] == 'one_hot'
    assert type(one_hot_transformer[1]) == sklearn.preprocessing._encoders.OneHotEncoder
    assert one_hot_transformer[2] == ['Education', 'City']
    # same tests on preprocessing_pipeline_linear
    assert one_hot_transformer_lin[0] == 'one_hot'
    assert type(one_hot_transformer_lin[1]) == sklearn.preprocessing._encoders.OneHotEncoder
    assert one_hot_transformer_lin[2] == ['Education', 'City']


def test_binary_encoding():
    """
    Test that the binary transformer is well stored (storage order).
    """
    binary_transformer = preprocessing_pipeline.transformers_[1]
    binary_transformer_lin = preprocessing_pipeline_linear.transformers_[1]

    assert binary_transformer[0] == 'to_binary'
    assert type(binary_transformer[1]) == sklearn.preprocessing._encoders.OrdinalEncoder
    assert binary_transformer[2] == ['Gender', 'EverBenched']
    # same tests on preprocessing_pipeline_linear
    assert binary_transformer_lin[0] == 'to_binary'
    assert type(binary_transformer_lin[1]) == sklearn.preprocessing._encoders.OrdinalEncoder
    assert binary_transformer_lin[2] == ['Gender', 'EverBenched']


def test_numerical_lin_pipeline():
    """
    Test that the numerical transformer is well stored (storage order).
    """
    num_transformer = preprocessing_pipeline_linear.transformers_[2]

    assert num_transformer[0] == 'numerical'
    assert type(num_transformer[1]) == sklearn.preprocessing._data.StandardScaler
    assert num_transformer[2] == ['JoiningYear', 'Age', 'ExperienceInCurrentDomain']


def test_preprocessing_function():
    """
    This function tests if the preprocessing step does what you want it to do.
    """
    # loads train data
    train_data_path = os.path.join('data', 'training', 'x_train.csv')
    X_train = pd.read_csv(train_data_path)
    usecols = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender',
       'EverBenched', 'ExperienceInCurrentDomain']
    X_train = X_train[usecols]

    # prepare manually the data
    # one-hot encoder
    one_hot = OneHotEncoder()
    # categorial variables
    col_one_hot = ['Education', 'City']
    ord_binary = OrdinalEncoder()
    col_binary = ['Gender', 'EverBenched']
    std_scaler = StandardScaler()
    col_num = ['JoiningYear', 'Age', 'ExperienceInCurrentDomain']

    preprocessing_pipeline_test = ColumnTransformer([
        ("one_hot", one_hot, col_one_hot),
        ("to_binary", ord_binary, col_binary)
    ])

    preprocessing_pipeline_test_lin = ColumnTransformer([
        ("one_hot", one_hot, col_one_hot),
        ("to_binary", ord_binary, col_binary),
        ('num', std_scaler, col_num)
    ])

    prepared_col = preprocessing_pipeline_test.fit_transform(X_train)
    X_train_drop = X_train.drop(columns=col_one_hot + col_binary)
    X_prepared = np.concatenate((X_train_drop, prepared_col), axis=1)
    # same for linear pipeline
    prepared_col_lin = preprocessing_pipeline_test_lin.fit_transform(X_train)
    X_train_drop_lin = X_train.drop(columns=col_one_hot + col_binary + col_num)
    X_prepared_lin = np.concatenate((X_train_drop_lin, prepared_col_lin), axis=1)

    assert np.array_equal(
            prepare_raw_data(X_train, preprocessing_pipeline=preprocessing_pipeline),
            X_prepared)
    # test for linear case
    assert np.array_equal(
            prepare_raw_data(X_train, preprocessing_pipeline=preprocessing_pipeline_linear,
                             numerical_std=True),
            X_prepared_lin)