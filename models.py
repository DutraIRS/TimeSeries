from abc import ABC, abstractmethod


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
    
    def _evaluate(self, metric):
        ...
    
    def evaluate(self, metrics):
        report = {}
        
        for metric in metrics:
            self._evaluate(metric)
        
        return report


# Baseline models
class Mean(Model):
    def fit(self, X, y):
        self.mean = y.mean()

    def predict(self, X):
        return self.mean


class RandomWalk(Model):
    def fit(self, X, y):
        self.last_value = y[-1]
    def predict(self, X):
        return self.last_value


class SeasonalRandomWalk(Model):
    ...


class Drift(Model):
    ...


class ExponentialSmoothing(Model):
    ...


class LinearRegression(Model):
    ...


class SARIMA(Model):
    ...


class Conv1D(Model):
    ...


class LSTM(Model):
    ...