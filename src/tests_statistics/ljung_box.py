import numpy as np
from scipy.stats import chi2

def ljung_box_test(data, lags=10):
    """
    Implémente le test de Ljung-Box pour vérifier l'absence d'autocorrélation.
    
    Args:
        data (array-like): Les données (log-returns ou série temporelle).
        lags (int): Nombre maximal de lags à tester.
        
    Returns:
        dict: Résultats avec les valeurs de Q, les p-values, et la conclusion.
    """
    n = len(data)
    q_stat = 0
    autocorrelations = []

    # Calcul des autocorrélations pour chaque lag
    for lag in range(1, lags + 1):
        autocorr = np.corrcoef(data[:-lag], data[lag:])[0, 1]  # Coefficient d'autocorrélation
        autocorrelations.append(autocorr)
        q_stat += (autocorr ** 2) / (n - lag)

    # Ajuster la statistique Q
    q_stat = n * (n + 2) * q_stat

    # Calculer la p-value en utilisant la distribution chi-carré
    p_value = 1 - chi2.cdf(q_stat, df=lags)

    # Conclusion
    conclusion = "No significant autocorrelation" if p_value > 0.05 else "Significant autocorrelation"

    return {
        "Q-Statistic": q_stat,
        "p-value": p_value,
        "Conclusion": conclusion,
        "Autocorrelations": autocorrelations,
    }