"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    features = ['Madrid_weather_id', 'Barcelona_weather_id', 'Seville_weather_id', 'Bilbao_weather_id',
                'Madrid_clouds_all', 'Bilbao_clouds_all', 'Seville_clouds_all', 'Madrid_temp_min',
                'Seville_temp_min', 'Bilbao_temp_max', 'Barcelona_temp_min', 'Bilbao_temp_min', 'Madrid_temp_max',
                'Barcelona_temp_max', 'Valencia_temp_min', 'Valencia_temp_max', 'Seville_temp_max',
                'Barcelona_pressure', 'Valencia_snow_3h', 'Seville_rain_3h', 'Bilbao_snow_3h']

    def replace_null_with_mean(sr):
        new_sr = sr.copy()
        mean_val = sr.mean()
        new_sr = new_sr.fillna(mean_val)
        return new_sr

    def check_null(df):
        new_df = df.copy()
        cols = new_df.columns
        for col in cols:
            if new_df[col].isnull().sum() > 0:
                new_df[col] = replace_null_with_mean(new_df[col])
        return new_df

    def remove_features(df, cols):
        df_cols = df.columns
        remove_col = [col for col in cols if col in df_cols]
        new_df = df.drop(columns=remove_col, axis=1)
        return new_df

    def id_outliers(sr):
        data = sr.values
        q25 = sr.quantile(.25)
        q75 = sr.quantile(.75)
        q50 = sr.mean()
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        outliers = [x for x in data if x < lower or x > upper]
        return (outliers, q50)

    def handle_outliers(sr, outliers, value):
        sr_copy = sr.copy()
        sr_copy = sr_copy.replace(outliers, value)
        return sr_copy

    def treat_outliers(df):
        df_copy = df.copy()
        cols = df_copy.columns

        for col in cols:
            if ((df_copy[col].dtype != 'O') & (col != 'load_shortfall_3h')):
                outliers, median = id_outliers(df_copy[col])
                df_copy[col] = handle_outliers(df_copy[col], outliers, median)
        return df_copy

    def conver_time(df, col):
        return pd.to_datetime(df[col])

    def object_converter(df):
        new_df = df.copy()
        cols = df.columns
        object_type = [col for col in cols if df[col].dtype == 'O']
        new_df = pd.get_dummies(df, columns=object_type, drop_first=True)
        return new_df

    def split_datetime_col(df, col):
        new_df = df.copy()
        new_df['Year'] = new_df[col].dt.year
        new_df['Month'] = new_df[col].dt.month
        new_df['Week'] = new_df[col].dt.week
        new_df['Day'] = new_df[col].dt.day
        new_df['Hour'] = new_df[col].dt.hour
        return new_df


def standadize_data(df, exclude):
    original_columns = df.columns
    df_copy = df.copy()
    df_copy = df_copy.drop(exclude, axis=1)
    std_columns = df_copy.columns
    # create scaler object
    scaler = StandardScaler()

    df_copy_scaled = scaler.fit_transform(df_copy)
    df_copy_scaled = pd.DataFrame(df_copy_scaled, columns=std_columns)

    for col in exclude:
        df_copy_scaled[col] = df[col]

    df_copy_scaled = df_copy_scaled[original_columns]
    return df_copy_scaled

    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    # Team_1 Added this
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    # ------------------------------------------------------------------------

    clean_test_df = check_null(feature_vector_df)
    clean_test_df = remove_features(clean_test_df, features)
    clean_test_df = treat_outliers(clean_test_df)
    clean_test_df['time'] = conver_time(clean_test_df, 'time')
    clean_test_df_copy = object_converter(clean_test_df)
    clean_test_df_copy = split_datetime_col(clean_test_df_copy, 'time')
    predict_test_data_time = clean_test_df['time']
    predict_test_data_time = predict_test_data_time.to_numpy()
    clean_test_df_copy = clean_test_df_copy.drop('time', axis=1)
    predict_vector = standadize_data(clean_test_df_copy, [])

    return predict_vector


def load_model(path_to_model: str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""


def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
