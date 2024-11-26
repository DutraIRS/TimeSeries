import models
import metrics
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

def plot_prediction(models_list, X, y):
    """
    Plots the predictions of the given models on the given data.
    """
    # Create a grid of plots
    num_plots = len(models_list)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
    
    # Plot the data in every subplot
    for ax in axes:
        ax.plot(y, label='Consumption', color='black', alpha=0.6, linestyle='--')
    
    # Plot the predictions
    for i, model in enumerate(models_list):
        model.fit(X, y)
        y_pred = model.predict(X)
        axes[i].plot(y_pred, label=str(model), color='red', alpha=0.6)
        axes[i].set_title(str(model))
    
    plt.legend()
    plt.show()


def main():
    # Load data
    data = pd.read_csv("us_change_cleaned.csv").drop(columns='Date')
    X = data.drop(columns='Consumption')
    y = data['Consumption']
    
    X = (X - X.mean()) / X.std()
    # Define models
    models_list = [
        models.Mean(),
        models.RandomWalk(),
        models.SeasonalRandomWalk(2),
        models.SeasonalRandomWalk(3),
        models.SeasonalRandomWalk(4),
        models.Drift(),
        models.ExponentialSmoothing(1),
        models.LinearRegression(),
        models.SARIMA((1, 0, 1), (1, 0, 1, 4)),
        models.Conv1DModel(1, 4, 5),
        models.LSTMModel(5, 1)
    ]
    
    # Define metrics to evaluate goodness of fit
    fit_metrics_list = [
        metrics.r_squared,
        metrics.aic,
        metrics.aic_corrected,
        metrics.bic
    ]
    
    # Define metrics to evaluate predictive capacity
    pred_metrics_list = [
        metrics.rmse,
        metrics.mae,
        metrics.mape
    ]
    
    # Cross-validation splitter
    num_splits = 5
    ts_cv = TimeSeriesSplit(gap=0, n_splits=num_splits)
    
    fit_results = {}
    pred_results = {}
    for model in models_list:
        model_results = {}
        
        # Goodness of fit evaluation
        model.fit(X, y)
        y_pred = model.predict(X)
        
        for metric in fit_metrics_list:
            model_results[metric.__name__] = metric(y, y_pred, model.num_params())
        
        fit_results[str(model)] = model_results
        
        model_results = {}
        
        # Cross-validation loop to evaluate predictive capacity
        folds_results = np.zeros((num_splits, len(pred_metrics_list)))
        for i, (train_index, test_index) in enumerate(ts_cv.split(X)):
            # Fit and predict fold data
            model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = model.predict(X.iloc[test_index])
            
            # Evaluate predictions
            for j, metric in enumerate(pred_metrics_list):
                folds_results[i, j] = metric(y.iloc[test_index], y_pred, model.num_params())
            
        # Average results over folds
        cv_results = folds_results.mean(axis=0)
        
        for i, metric in enumerate(pred_metrics_list):
            model_results[metric.__name__] = cv_results[i]
        
        pred_results[str(model)] = model_results
        
    print(pd.DataFrame(fit_results).T)
    print(pd.DataFrame(pred_results).T)


if __name__ == "__main__":
    main()