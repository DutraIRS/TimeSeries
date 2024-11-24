import numpy as np
from abc import ABC, abstractmethod
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, LSTM, Input


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
    def __init__(self, seasonality):
        self.seasonality = seasonality
    
    def fit(self, X, y):
        self.last_observation = X.iloc[-1]
        self.last_season = y.iloc[-self.seasonality:].values
    
    def predict(self, X):
        h = X['Quarter'] - self.last_observation['Quarter']
        k = np.floor((h - 1) / self.seasonality).astype(int)
        
        indices = (h - self.seasonality * (k + 1)) - 1
        indices = indices.astype(int)
        
        return self.last_season[indices]

    def num_params(self):
        return self.seasonality


class Drift(Model):
    """
    Model that predicts the trend of the target variable.
    """
    def fit(self, X, y):
        self.C = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
        self.last_observation = X.iloc[-1]
        self.last_value = y.iloc[-1]
    
    def predict(self, X):
        self.h = X['Quarter'] - self.last_observation['Quarter']
        return self.last_value + self.C * self.h
    
    def num_params(self):
        return 2


# Non-baseline models
class ExponentialSmoothing(Model):
    """
    Model that predicts based on a smoothed moving average of the target variable.
    """
    def __init__(self, alpha):
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        
        self.alpha = alpha
    
    def fit(self, X, y):
        # Initialize the smoothed value with the first observation
        self.smoothed_values = np.zeros(len(y))
        self.smoothed_values[0] = y.iloc[0]
        
        # Apply the exponential smoothing formula
        for t in range(1, len(y)):
            self.smoothed_values[t] = self.alpha * y.iloc[t] + (1 - self.alpha) * self.smoothed_values[t - 1]
    
    def predict(self, X):
        if len(self.smoothed_values) == 0:
            raise ValueError("Model must be fitted before predicting.")
        
        # The simplest form of forecasting with exponential smoothing
        last_smoothed = self.smoothed_values[-1]
        
        return np.array([last_smoothed] * len(X))
    
    def num_params(self):
        # The only parameter is alpha (the smoothing constant)
        return 1


class LinearRegression(Model):
    """
    Model that predicts the target variable using a linear combination of the features.
    """
    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X.values]
        self.coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y.values

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X.values]  
        return X @ self.coefficients
    
    def num_params(self):
        return len(self.coefficients)


class SARIMA(Model):
    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None

    def fit(self, X, y):
        # SARIMA = SARIMAX without exogenous variables
        self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
        
        self.fitted_model = self.model.fit(disp=False)
    
    def predict(self, X):
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before predicting.")
        
        return self.fitted_model.forecast(steps=len(X))
    
    def num_params(self):
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before getting the number of parameters.")
        
        return len(self.fitted_model.params)


class Conv1DModel(Model):
    def __init__(self, filters, kernel_size, input_size):
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_size = input_size
        
        self.model = Sequential()
        self.model.add(Input(shape=(self.input_size, 1)))
        self.model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                            activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def fit(self, X, y, epochs=50, batch_size=8):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def predict(self, X):
        return self.model.predict(X).flatten()
    
    def num_params(self):
        return self.model.count_params()


class LSTMModel(Model):
    def __init__(self, units, input_size):
        self.units = units
        self.input_size = input_size
        
        self.model = Sequential()
        self.model.add(Input(shape=(self.input_size, 1)))
        self.model.add(LSTM(units=self.units, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def fit(self, X, y, epochs=50, batch_size=8):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def predict(self, X):
        return self.model.predict(X).flatten()
    
    def num_params(self):
        return self.model.count_params()



