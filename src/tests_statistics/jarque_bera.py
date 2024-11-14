from scipy.stats import chi2

from tests_statistics.moments import empirical_kurtosis, empirical_skewness

def jarque_bera_test(data, column):
    """
    Effectue le test de Jarque-Bera pour la normalité sur une colonne donnée.
    
    Args:
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne sur laquelle effectuer le test.
        
    Returns:
        dict: Un dictionnaire avec la statistique de test JB, la p-value, 
              et une indication de normalité.
    """
    # Calcul de l'asymétrie et de la kurtose empiriques
    skewness = empirical_skewness(data, column)
    kurtosis = empirical_kurtosis(data, column)
    n = len(data[column].dropna())
    
    # Calcul de la statistique de test de Jarque-Bera
    jb_statistic = (n / 6) * (skewness**2 + ((kurtosis - 3)**2) / 4)
    
    # Calcul de la p-value basée sur une distribution Chi-carré avec 2 degrés de liberté
    p_value = 1 - chi2.cdf(jb_statistic, df=2)
    
    # Interprétation du test
    is_normal = p_value > 0.05  # Hypothèse nulle de normalité rejetée si p < 0.05
    
    return {
        'JB Statistic': jb_statistic,
        'p-value': p_value,
        'Normality': 'Accepted' if is_normal else 'Rejected'
    }