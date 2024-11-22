import models
import numpy as np
import pandas as pd


def main():
    data = pd.read_csv("us_change_cleaned.csv")
    X = data.drop(columns='Consumption')
    y = data['Consumption']

    X_train = X[:int(0.8*len(X))]
    y_train = y[:int(0.8*len(y))]
    X_test = X[int(0.8*len(X)):]
    y_test = y[int(0.8*len(y)):]
    
    models_list = [
        models.Mean(),
        models.RandomWalk(),
        models.SeasonalRandomWalk(),
        models.Drift(),
        models.ExponentialSmoothing(),
        models.LinearRegression(),
        models.SARIMA(),
        models.Conv1D(),
        models.LSTM(),
    ]
    
    metrics_list = [
        ...
    ]
    
    for model in models_list:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {}
        for metric in metrics_list:
            metrics[metric.__name__] = metric(y_test, y_pred)
        
        


if __name__ == "__main__":
    main()