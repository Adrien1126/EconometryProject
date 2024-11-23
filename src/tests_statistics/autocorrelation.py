import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def autocovariance(data, lag):
    """
    Calcule l'autocovariance pour un lag donné.
    
    Args:
        data (array-like): La série temporelle des log-returns.
        lag (int): Le décalage pour lequel calculer l'autocovariance.
        
    Returns:
        float: La valeur de l'autocovariance au lag donné.
    """
    n = len(data)
    mean = np.mean(data)  # Moyenne de la série
    cov = np.sum((data[:n-lag] - mean) * (data[lag:] - mean)) / n
    return cov

def autocorrelation(data, lag):
    """
    Calcule l'autocorrélation pour un lag donné.
    
    Args:
        data (array-like): La série temporelle des log-returns.
        lag (int): Le décalage pour lequel calculer l'autocorrélation.
        
    Returns:
        float: La valeur de l'autocorrélation au lag donné.
    """
    numerator = autocovariance(data, lag)
    denominator = autocovariance(data, 0)  # La variance est l'autocovariance au lag 0
    return numerator / denominator



def plot_autocorrelations(data, max_lag=20):
    """
    Trace les autocorrélations pour plusieurs lags.
    
    Args:
        data (array-like): La série temporelle des log-returns.
        max_lag (int): Nombre maximum de lags à tracer.
    """
    autocorrelations = [autocorrelation(data, lag) for lag in range(1, max_lag + 1)]
    
    plt.figure(figsize=(10, 6))
    plt.stem(range(1, max_lag + 1), autocorrelations, basefmt=" ", use_line_collection=True)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Autocorrelations")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.show()


def plot_autocorrelations_with_ci(data, max_lag=20, alpha=0.05):
    """
    Trace les autocorrélations avec intervalles de confiance.
    
    Args:
        data (array-like): La série temporelle des log-returns.
        max_lag (int): Nombre maximum de lags à tracer.
        alpha (float): Niveau de signification pour les intervalles de confiance.
    """
    n = len(data)
    z_alpha = norm.ppf(1 - alpha / 2)  # Valeur critique pour l'intervalle de confiance
    ci = z_alpha / np.sqrt(n)         # Intervalle de confiance
    
    autocorrelations = [autocorrelation(data, lag) for lag in range(1, max_lag + 1)]
    
    plt.figure(figsize=(10, 6))
    plt.stem(range(1, max_lag + 1), autocorrelations, basefmt=" ")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(ci, color='red', linestyle='--', linewidth=0.8, label="95% Confidence Interval")
    plt.axhline(-ci, color='red', linestyle='--', linewidth=0.8)
    plt.title("Autocorrelations with Confidence Intervals")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.show()


