import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - ss_res / ss_tot

# gepetado, conferir se ta certo
def aic(y_true, y_pred, num_params):
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    
    return n * np.log(rss / n) + 2 * num_params

# gepetado, conferir se ta certo
def bic(y_true, y_pred, num_params):
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    
    return n * np.log(rss / n) + num_params * np.log(n)