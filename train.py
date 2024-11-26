import models
import metrics
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import json
from sklearn.preprocessing import PowerTransformer

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


def train_tranformation(transform, inv_transform, archive_name):
    print("Init ", archive_name)
    # Load data
    data = pd.read_csv("us_change_cleaned.csv").drop(columns='Date')
    X = data.drop(columns='Consumption')
    y = data['Consumption']

    y_original = y.copy()

    X, y = transform(X, y)

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
    residuals_results = {}

    for model in models_list:
        model_results = {}
        
        # Goodness of fit evaluation
        model.fit(X, y)
        y_pred = model.predict(X)
        # Convert to numpy array in models that return core.Series
        y_pred = pd.Series(y_pred).to_numpy()

        # Invert predictions to measure metrics
        y_pred_inv = inv_transform(y_pred.reshape(-1,1)).reshape(-1)
        for metric in fit_metrics_list:
            model_results[metric.__name__] = metric(y_original, y_pred_inv, model.num_params())
        
        fit_results[str(model)] = model_results
        
        model_results = {}
        
        # Cross-validation loop to evaluate predictive capacity
        folds_results = np.zeros((num_splits, len(pred_metrics_list)))
        for i, (train_index, test_index) in enumerate(ts_cv.split(X)):
            # Fit and predict fold data
            model.fit(X.iloc[train_index], y[train_index])
            y_pred_cross = model.predict(X.iloc[test_index])

            # Convert to numpy array models that return core.Series
            y_pred_cross = pd.Series(y_pred_cross).to_numpy()

            # Invert predictions to measure metrics
            y_pred_cross_inv = inv_transform(y_pred_cross.reshape(-1,1)).reshape(-1)

            # Evaluate predictions
            for j, metric in enumerate(pred_metrics_list):
                folds_results[i, j] = metric(y_original[test_index], y_pred_cross_inv, model.num_params())

        # Average results over folds
        cv_results = folds_results.mean(axis=0)
        
        for i, metric in enumerate(pred_metrics_list):
            model_results[metric.__name__] = cv_results[i]
        
        pred_results[str(model)] = model_results
        residuals_results[str(model)] = list(y_pred_inv - y_original)
    
    with open("results/fit_results_" + archive_name + ".json", "w") as f:
        json.dump(pd.DataFrame(fit_results).T.to_dict(), f, indent=4)
    
    with open("results/pred_results_" + archive_name + ".json", "w") as f:
        json.dump(pd.DataFrame(pred_results).T.to_dict(), f, indent=4)

    with open("results/residual_" + archive_name + ".json", "w") as f:
        json.dump(residuals_results, f, indent=4)

    print(archive_name + " Done!")

def main():
    def null_transform(X, y):
        return X, y
    
    def null_inv_tranform(y):
        return y
    
    def normalization(X, y):
        quarter = X["Quarter"]
        X = X.drop(columns="Quarter")

        X = (X - X.mean()) / X.std()

        X["Quarter"] = quarter
        return X, y
    

    power_transform = PowerTransformer(method='yeo-johnson')
    def norm_power_transform(X, y):
        # Normalization
        quarter = X["Quarter"]
        X = X.drop(columns="Quarter")

        X = (X - X.mean()) / X.std()

        X["Quarter"] = quarter

        y = power_transform.fit_transform(y.values.reshape(-1,1))
        y = pd.Series(y.reshape(-1))
        return X, y
    
    def inv_power_transform(y):
        return power_transform.inverse_transform(y.reshape(-1,1)).reshape(-1)
    
    train_tranformation(null_transform, null_inv_tranform, "sem_transformacao")
    train_tranformation(normalization, null_inv_tranform, "normalizacao")
    train_tranformation(norm_power_transform, inv_power_transform, "power_transform")


if __name__ == "__main__":
    main()