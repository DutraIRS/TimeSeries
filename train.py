import models
import metrics
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


def main():
    # Load data
    data = pd.read_csv("us_change_cleaned.csv").drop(columns='Date')
    X = data.drop(columns='Consumption')
    y = data['Consumption']
    
    # Define models
    models_list = [
        models.Mean(),
        models.RandomWalk(),
        models.SeasonalRandomWalk(4),
        models.Drift(),
        models.ExponentialSmoothing(1),
        models.LinearRegression(),
        models.SARIMA((1, 0, 1), (1, 0, 1, 4)),
        models.Conv1DModel(1, 4, 5),
        models.LSTMModel(5, 1)
    ]
    
    # Define metrics
    metrics_list = [
        metrics.rmse,
        metrics.mae,
        metrics.mape,
        metrics.r_squared,
        metrics.aic,
        metrics.aic_corrected,
        metrics.bic
    ]
    
    # Cross-validation splitter
    num_splits = 10
    ts_cv = TimeSeriesSplit(gap=0, n_splits=num_splits)
    
    # Cross-validation loop
    results = {}
    for model in models_list:
        model_results = {}
        
        folds_results = np.zeros((num_splits, len(metrics_list)))
        for i, (train_index, test_index) in enumerate(ts_cv.split(X)):
            # Fit and predict fold data
            model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = model.predict(X.iloc[test_index])
            
            # Evaluate predictions
            for j, metric in enumerate(metrics_list):
                folds_results[i, j] = metric(y.iloc[test_index], y_pred, model.num_params())
            
        # Average results over folds
        cv_results = folds_results.mean(axis=0)
        
        for i, metric in enumerate(metrics_list):
            model_results[metric.__name__] = cv_results[i]
        
        results[model.__class__.__name__] = model_results
        
    print(pd.DataFrame(results).T)


if __name__ == "__main__":
    main()