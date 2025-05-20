import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import root_mean_squared_error, r2_score
from typing import Literal
import os

sns.set_theme()

# Some predefined functions which will be moved to utils.py file.

# %%
def create_dataset(df, window=1, predicted_interval=1, fillna=False, size: None|int=None) -> tuple:
    """
    Create a dataset for time series forecasting by getting the previous window
    time steps as input features and the next predicted_interval time steps
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        window (int): Number of previous time steps to use as input features.
        predicted_interval (int): Number of time steps to predict.
        fillna (bool): Whether to fill NaN values in the dataset by interpolation with 'time' method.
        size (int, optional): If specified, get the last `size` rows from the dataset.
    Returns:
        tuple: A tuple containing the input features (X) and target variable (y).
        X (pd.DataFrame): Input features for the model.
        y (pd.Series): Target variable for the model.
    """

    X = df[['Close']]
    X = X.asfreq('D')
    X['Target'] = X['Close'].shift(-predicted_interval)

    if fillna:
        X['Close'] = X['Close'].interpolate(method='time')

    for i in range(window):
        X[f'Lag_{i}'] = X['Close'].shift(i)     

    if size and len(X) > size:
        X = X.tail(size)
    X = X.dropna()
    X.drop(columns=['Close'], inplace=True)
    y = X.pop('Target')
    
    return X, y

# %%
def create_custom_dataset(df, window=2, predicted_interval=1, fillna=False, size: None|int=None) -> tuple:
    X, y = create_dataset(df, window=window, predicted_interval=predicted_interval, fillna=fillna, size=size)
    X['lag_2'] = X['lag_1'] - X['lag_2']
    X.rename(columns={'lag_2': 'diff'}, inplace=True)
    X['mon_fri'] = ((y.index + pd.Timedelta(days=predicted_interval)).weekday.isin([0, 4])).astype(int)
    
    return X, y

# %%
def split_dataset(X, y, test_size=0.2, method: Literal['half', 'cv']='half'):
    """
    Split the dataset into training and testing sets.
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of test set to the dataset.
        method (str): ['half', 'cv'] Method for splitting the dataset. \n
        'half' - Split the dataset 2 parts. \n
        'cv' - Split the dataset into k folds for cross-validation.
        Returns:
        tuple: 
        X_train, X_test, y_train, y_test DataFrames if method='half'.
        X_trains, X_tests, y_trains, y_tests lists of DataFrames if method='cv'.
    """
    if test_size <= 0 or test_size >= 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.\n")

    if method == 'half':
        mid = int(len(X) * (1 - test_size))
        X_train, X_test = X[:mid], X[mid:]
        y_train, y_test = y[:mid], y[mid:]
        return X_train, X_test, y_train, y_test
    elif method == 'cv':
        folds = int(1 / test_size)
        start = 0
        end = int(len(X) * test_size)
        fold_size = int(len(X) * test_size)
        X_trains, X_tests, y_trains, y_tests = [], [], [], []
        for i in range(folds):
            X_trains.append(X[:end])
            y_trains.append(y[:end])
            X_tests.append(X[start:end])
            y_tests.append(y[start:end])
            start = end
            end = min(end + fold_size, len(X))
        X_tests = X_tests[1:]
        y_tests = y_tests[1:]
        X_trains = X_trains[:-1]
        y_trains = y_trains[:-1]
        return X_trains, X_tests, y_trains, y_tests

# %%
def APE(y_test: pd.Series, y_pred: pd.Series) -> np.ndarray:
    np_y_test = y_test.to_numpy()
    np_y_pred = y_pred.to_numpy()
    return np.abs((np_y_test - np_y_pred) / np_y_test) * 100

def MAPE(y_test: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    Args:
        y_test (pd.Series): True values.
        y_pred (pd.Series): Predicted values.
    Returns:
        float: MAPE value.
    """
    return np.mean(APE(y_test, y_pred))

def sMAPE(y_test: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    Args:
        y_test (pd.Series): True values.
        y_pred (pd.Series): Predicted values.
    Returns:
        float: sMAPE value.
    """
    np_y_test = y_test.to_numpy()
    np_y_pred = y_pred.to_numpy()
    return np.mean(np.abs(np_y_test - np_y_pred) / ((np.abs(np_y_test) + np.abs(np_y_pred)) / 2)) * 100

# %%
def full_model_plot(compared_df, model_name="model") -> None:
    """
    Plot the results of the model as 3 figures:
    1. Predicted vs Actual
    2. Average Percentage Error
    3. Combined of 1 and 2
    Args:
        compared_df (pd.DataFrame): df containing True and Predicted values.
        model_name (str): Name of the model.
    Returns:
        None
    """

    # Set up the color palette
    palette_name = 'viridis'
    n_colors = 3
    palette = sns.color_palette(palette_name, n_colors)
    color_predict = palette[0]
    color_true = 'coral'
    color_error = palette[2]

    # Main plot (y_test vs y_pred)
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=compared_df.index, y=compared_df['True'], label='True Values', color=color_true)
    sns.lineplot(x=compared_df.index, y=compared_df['Predicted'], label='Predicted Values', color=color_predict)
    plt.title(f'{model_name} Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Combined plot (True vs Predicted and APE)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(f'{model_name} Predictions and APE')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Stock Price', color=color_true)
    sns.lineplot(x=compared_df.index, y=compared_df['True'], label='True Values', color=color_true)
    sns.lineplot(x=compared_df.index, y=compared_df['Predicted'], label='Predicted Values', color=color_predict)
    ax1.tick_params(axis='y', labelcolor=color_true)
    ax1.legend(loc='upper left')
    ax1.grid(False)

    # Create a twin y-axis for APE
    ax2 = ax1.twinx()
    ape = APE(compared_df['True'], compared_df['Predicted'])
    ax2.set_ylabel('APE (%)', color=color_error)
    # sns.lineplot(x=compared_df.index, y=pd.Series(ape), label='APE', ax=ax2, color=color_error, linestyle='--')
    ape_series = pd.Series(ape, index=compared_df.index)
    for date, _ in compared_df.iterrows():
        ax2.plot(
            [date, date],
            [0, ape_series[date]],
            color=color_error,
            alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color_error)
    ax2.legend([plt.Line2D([0], [0], color=color_error)], ['APE'], loc='upper right')
    ax2.grid(False)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    fig.tight_layout()
    plt.show()

# %%
def get_result(pipeline, compared_df) -> dict:
    """
    Get the result of the model.
    Args:
        model (sklearn model): Trained model.
        compared_df (pd.DataFrame): DataFrame containing True and Predicted values.
    Returns:
        dict: key=['Result', 'Pipeline', 'Coef', 'Intercept', 'RMSE', 'R^2', 'MAPE', 'sMAPE']
    """
    
    rmse = root_mean_squared_error(compared_df['True'], compared_df['Predicted'])
    r2 = r2_score(compared_df['True'], compared_df['Predicted'])
    mape= MAPE(compared_df['True'], compared_df['Predicted'])
    smape = sMAPE(compared_df['True'], compared_df['Predicted'])
    
    return {
        'Result': compared_df,
        'Pipeline': pipeline,
        'Coef': pipeline.regressor_.named_steps['model'].coef_,
        'Intercept': pipeline.regressor_.named_steps['model'].intercept_,
        'RMSE': rmse,
        'R^2': r2,
        'MAPE': mape,
        'sMAPE': smape,
    }

# %%
def get_compared_df(pipeline, X, y) -> pd.DataFrame:
    """
    Get the compared DataFrame.
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): True values.
    Returns:
        pd.DataFrame: (True, Predicted).
    """
    
    return pd.DataFrame({
        'True': y, 
        'Predicted': pd.Series(pipeline.predict(X).flatten(), index=X.index)
        }, y.index)

# %%
def model_frame(model, X_train, X_test, y_train, y_test, show_plot=True, train_size=None) -> dict:
    """
    Train a model and evaluate its performance.
    Args:
        model: The machine learning model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target variable.
        show_plot (bool): Whether to show the plot of the model's predictions.
    Returns:
        dict: Dict containing the model's predictions and performance metrics.
    """
    # pipeline = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('model', model)
    # ])

    pipeline = TransformedTargetRegressor(
        regressor=Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ]),
        transformer=StandardScaler()
    )

    if train_size and train_size < len(X_train):
        X_train = X_train.tail(train_size)
        y_train = y_train.tail(train_size)
        
    pipeline.fit(X_train, y_train)

    compared_df = get_compared_df(pipeline, X_test, y_test)
    result = get_result(pipeline, compared_df)

    print(f'{type(model).__name__} RMSE: {result["RMSE"]:.4f}, R^2: {result["R^2"]:.4f}\n')
    print(f'Coef: {result["Coef"]}')
    print(f'MAPE: {result["MAPE"]:.4f}%')
    print(f'Max APE: {np.max(APE(result["Result"]["True"], result["Result"]["Predicted"])):.4f}%')
    print(f'sMAPE: {result["sMAPE"]:.4f}%')
    print()

    if show_plot:
        # model_X = pd.concat([X_train, X_test])
        # model_y = pd.concat([y_train, y_test])
        # full_compared_df = get_compared_df(pipeline, model_X, model_y)
        # full_model_plot(full_compared_df, type(model).__name__)
        full_model_plot(compared_df, type(model).__name__)
    
    return result

# %%
def cv_model_frame(model, X_trains, X_tests, y_trains, y_tests, show_plot=True, train_size=None) -> dict:
    """
    Iteratively take 1 fold for train and the next fold for test.
    Find average results.
    Args:
        model: The machine learning model to train. *e.g. LinearRegression()*
        X_trains (list): List of training features for each fold.
        y_trains (list): List of training target variable for each fold.
        X_tests (list): List of testing features for each fold.
        y_tests (list): List of testing target variable for each fold.
    Return:
        dict:
        'Result' (list): List of each fold's result.
        'AvgRMSE' (float)
        'AvgR^2' (float)
        'AvgMAPE' (float)
        'AvgsMAPE' (float)
    """
    results = {
        'AllResults' : [],
        'AvgRMSE' : None,
        'AvgR^2' : None,
        'AvgMAPE' : None,
        'AvgsMAPE' : None
    }

    for i in range(len(X_trains)):
        print(f'Fold {i + 1}/{len(X_trains)}')
        result = model_frame(model, X_trains[i], X_tests[i], y_trains[i], y_tests[i], show_plot=show_plot, train_size=train_size)
        results['AllResults'].append(result)
    
    results['AvgRMSE'] = np.mean([result['RMSE'] for result in results['AllResults']])
    results['AvgR^2'] = np.mean([result['R^2'] for result in results['AllResults']])
    results['AvgMAPE'] = np.mean([result['MAPE'] for result in results['AllResults']])
    results['AvgsMAPE'] = np.mean([result['sMAPE'] for result in results['AllResults']])
    results['StdRMSE'] = np.std([result['RMSE'] for result in results['AllResults']])
    results['StdR^2'] = np.std([result['R^2'] for result in results['AllResults']])
    results['StdMAPE'] = np.std([result['MAPE'] for result in results['AllResults']])
    results['StdsMAPE'] = np.std([result['sMAPE'] for result in results['AllResults']])
    print(f'Avg RMSE: {results["AvgRMSE"]:.4f} +/- {results["StdRMSE"]:.4f}')
    print(f'Avg R^2: {results["AvgR^2"]:.4f} +/- {results["StdR^2"]:.4f}')
    print(f'Avg MAPE: {results["AvgMAPE"]:.4f}% +/- {results["StdMAPE"]:.4f}%')
    print(f'Avg sMAPE: {results["AvgsMAPE"]:.4f}% +/- {results["StdsMAPE"]:.4f}%')
    
    return results

# %%
def finetune_frame(model, X_train, X_test, y_train, y_test, param_grid, cv=10) -> pd.DataFrame:
    """
    Fine-tune the model using GridSearchCV.
    Args:
        model: The machine learning model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training target variable.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.DataFrame): Testing target variable.
        param_grid (dict): param_grid of sklearn.GridSearchCV
        cv: Number of folds
    Returns:
        dict: Dict containing the model's predictions and performance metrics.
    """
    pipeline = TransformedTargetRegressor(
        regressor=Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ]),
        transformer=StandardScaler()
    )

    data = list(range(len(X_train)))
    k, m = divmod(len(data), cv)
    temp = [data[i*k + min(i, m) : (i+1)*k + min(i+1, m)] for i in range(cv)]
    custom_folds = [(list(range(t[0])), t) for t in temp[1:]]

    grid_search = GridSearchCV(pipeline, param_grid, cv=custom_folds, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_

    compared_df = get_compared_df(best_pipeline, X_test, y_test)
    result = get_result(best_pipeline, compared_df)
    result['GridSearch'] = grid_search

    print(f'{type(model).__name__} RMSE: {result["RMSE"]:.4f}, R^2: {result["R^2"]:.4f}\n')
    print(f'Coef: {result["Coef"]}')
    print(f'MAPE: {result["MAPE"]:.4f}%')
    print(f'Max APE: {np.max(APE(result["Result"]["True"], result["Result"]["Predicted"])):.4f}%')
    print(f'sMAPE: {result["sMAPE"]:.4f}%')

    return result

print("Utils loaded")