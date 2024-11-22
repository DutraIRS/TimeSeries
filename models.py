from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def num_params(self):
        """Returns the number of parameters of the model.
        """
        pass
    
    def _evaluate(self, metric, y_true, y_pred):
        if metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def evaluate(self, metrics, X, y):
        y_pred = self.predict(X)
        report = {}
        
        for metric in metrics:
            report[metric] = self._evaluate(metric, y, y_pred)
        
        return report


# Baseline models
class Mean(Model):
    def fit(self, X, y):
        self.mean = y.mean()

    def predict(self, x):
        return np.full(len(x), self.mean)
    
    def num_params(self):
        return 1


class RandomWalk(Model):
    def fit(self, X, y):
        self.last_value = y.iloc[-1]
    def predict(self, x):
        return np.full(len(x), self.last_value)
    def num_params(self):
        return 1


class SeasonalRandomWalk(Model):
    def fit(self, X, y,seasonality):
        self.seasonality = seasonality
        self.last_season = y.iloc[-seasonality:]
        self.season_orientation = X.iloc[-seasonality:]['Quarter'] % seasonality
    def predict(self, x):
        quarters = x['Quarter']
        quarters = quarters % self.seasonality
        # Print the position index for the values of quarters in the season_orientation
        predictions_index = np.array([np.where(self.season_orientation == quarter)[0][0] for quarter in quarters])
        predictions = []
        for i in range(len(predictions_index)):
            location = predictions_index[i]
            predictions.append(self.last_season.iloc[location])
        return predictions
    def num_params(self):
        return self.seasonality



class Drift(Model):
    ...


class ExponentialSmoothing(Model):
    ...


class LinearRegression(Model):
    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X.values]
        self.coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y.values

    def predict(self, x):
        x = np.c_[np.ones(x.shape[0]), x.values]  
        return x @ self.coefficients
    
    def num_params(self):
        return len(self.coefficients)


class SARIMA(Model):
    ...


class Conv1D(Model):
    ...


class LSTM(Model):
    ...



