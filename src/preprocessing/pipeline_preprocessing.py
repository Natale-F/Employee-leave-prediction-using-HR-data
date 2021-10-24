import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer


def prepare_raw_data(df: pd.DataFrame, preprocessing_pipeline: ColumnTransformer,
                     numerical_std: bool = False):
    """
    This function prepares the raw data and returns a matrix ready for use by a Machine Learning algorithm.

    Args:
        df (pd.DataFrame): raw data in a dataframe
        preprocessing_pipeline (sklearn.compose.ColumnTransformer): the object that contains the preprocessing steps
        numerical_std (bool): if the pipeline contains numerical steps

    Returns:
        prepared_data (np.ndarray): prepared data in a numpy matrix

    """
    # recovery the names of the columns on which we work:
    col_one_hot = preprocessing_pipeline.transformers_[0][2]
    col_binary = preprocessing_pipeline.transformers_[1][2]

    if numerical_std:
        col_num = preprocessing_pipeline.transformers_[2][2]
        preprocessed_columns = col_binary + col_one_hot + col_num
    else:
        preprocessed_columns = col_binary + col_one_hot

    # use the preprocessing pipeline
    prepared_col = preprocessing_pipeline.transform(df)
    # remove preprocessed columns
    df_for_prep = df.drop(columns=preprocessed_columns)
    # concatenate on the columns
    X_prepared = np.concatenate((df_for_prep, prepared_col), axis=1)
    return X_prepared


