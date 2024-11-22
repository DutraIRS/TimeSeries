from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    """
    Base class for models. All models must implement fit, predict, and num_params methods.
    """
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def num_params(self):
        """
        Returns the number of parameters of the model.
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
    """
    Model that predicts the mean of the target variable.
    """
    def fit(self, X, y):
        self.mean = np.mean(y)

    def predict(self, X):
        return np.full(len(X), self.mean)
    
    def num_params(self):
        return 1


class RandomWalk(Model):
    """
    Model that predicts the last observed value of the target variable.
    """
    def fit(self, X, y):
        self.last_value = y.iloc[-1]

    def predict(self, X):
        return np.full(len(X), self.last_value)

    def num_params(self):
        return 1


class SeasonalRandomWalk(Model):
    """
    Model that predicts based on seasonal patterns of the target variable.
    """
    def fit(self, X, y, seasonality):
        self.seasonality = seasonality
        self.last_observation = X.iloc[-1]
        self.last_season = y.iloc[-seasonality:].values
    def predict(self, x):
        h = x['Quarter'] - self.last_observation['Quarter']
        k = np.floor((h - 1) / self.seasonality).astype(int)
        indices = (h - self.seasonality * (k + 1)) -1
        return self.last_season[indices]

    def num_params(self):
        return self.seasonality



class Drift(Model):
    """
    Model that predicts the trend of the target variable.
    """
    def fit(self, X, y):
        self.C = (y.iloc[-1] - y.iloc[0]) / (len(y)-1)
        self.last_value = y.iloc[-1]
        self.last_observation = X.iloc[-1]
    def predict(self, x):
        self.h = x['Quarter'] - self.last_observation['Quarter']
        return self.last_value + self.C * self.h
    def num_params(self):
        return 2


class ExponentialSmoothing(Model):
    def fit(self,X,y,alpha):
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = alpha
        self.initial_value = y[0]  
        self.smoothed_values = np.zeros(len(y))
        self.smoothed_values[0] = self.initial_value
        for t in range(1, len(y)):
            self.smoothed_values[t] = self.alpha * y[t] + (1 - self.alpha) * self.smoothed_values[t - 1]

    def predict(self,x,steps):
        if self.smoothed_values is None:
            raise ValueError("Model must be fitted before predicting.")
        last_smoothed = self.smoothed_values[-1]
        return [last_smoothed] * steps
    
    def evaluate(self, metrics, X, y, steps):
        y_pred = self.predict(X, steps)
        y_true = y[-steps:]
        report = {}
        
        for metric in metrics:
            report[metric] = self._evaluate(metric, y_true, y_pred)
        
        return report
    
    def num_params(self):
        return 1

class LinearRegression(Model):
    """
    Model that predicts the target variable using a linear combination of the features.
    """
    def fit(self, X, y):
        X = X.drop(columns=["Date"])
        X = np.c_[np.ones(X.shape[0]), X.values]
        self.coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y.values

    def predict(self, x):
        x = x.drop(columns=["Date"])
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



