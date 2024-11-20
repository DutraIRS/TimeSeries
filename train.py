import models
import numpy as np
import pandas as pd


def main():
    data = pd.read_csv("us_change.csv")
    
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
        model.fit(...)
        


if __name__ == "__main__":
    main()