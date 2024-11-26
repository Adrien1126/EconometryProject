import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def empirical_var(data, alpha, isnormal=False):
    """
    Estimate VaR using either the empirical quantile method or the normal distribution.

    Args:
        data (array-like): The historical returns data.
        alpha (float): The confidence level (e.g., 0.99 for 99%).
        isnormal (bool): If True, assume a normal distribution to compute VaR.

    Returns:
        float: VaR estimate.
    """
    if isnormal:
        # Compute VaR for normal distribution
        mean = np.mean(data)
        std_dev = np.std(data)
        z = norm.ppf(1 - alpha)  # z-score for the confidence level
        return mean + z * std_dev
    else:
        # Compute empirical quantile VaR
        sorted_data = np.sort(data)
        n = len(data)
        m = int(np.floor(n * (1 - alpha)))  # Compute the rank m for 1 - alpha
        return sorted_data[m]

def hill_estimator(data, threshold):
    """
    Estimate the tail index (gamma) using the Hill estimator.

    Args:
        data (array-like): The historical returns data.
        threshold (float): The threshold (u) for extreme values.

    Returns:
        float: Tail index (gamma_hat).
    """
    tail_data = data[data < threshold]
    n_u = len(tail_data)
    if n_u == 0:
        raise ValueError("No data below the threshold!")
    
    log_diffs = np.log(-tail_data) - np.log(-threshold)
    gamma_hat = 1 / (np.sum(log_diffs) / n_u)
    return gamma_hat, n_u

def pareto_var(data, alpha, threshold):
    """
    Estimate VaR using the Pareto-type model and Hill estimator.

    Args:
        data (array-like): The historical returns data.
        alpha (float): The confidence level (e.g., 0.99 for 99%).
        threshold (float): The threshold (u) for extreme values.

    Returns:
        float: VaR estimate.
    """
    gamma_hat, n_u = hill_estimator(data, threshold)
    n = len(data)
    VaR = -threshold * (n_u / (n * alpha))**(1 / gamma_hat)
    return VaR



def plot_empirical_distribution(data, alpha, VaR):
    """
    Plot the empirical distribution and mark the VaR estimate.

    Args:
        data (array-like): The historical returns data.
        alpha (float): The confidence level (e.g., 0.99 for 99%).
        VaR (float): The VaR estimate.
    """
    sorted_data = np.sort(data)
    n = len(data)
    ecdf = np.arange(1, n + 1) / n

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, ecdf, label="Empirical Distribution Function")
    plt.axvline(x=VaR, color='red', linestyle='--', label=f"VaR ({alpha*100:.1f}%) = {VaR:.4f}")
    plt.title("Empirical Distribution Function and VaR")
    plt.xlabel("Returns")
    plt.ylabel("F(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

