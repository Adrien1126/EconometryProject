import pandas as pd
import numpy as np

def empirical_mean(data, column):
    """
    Calcule la moyenne empirique pour une colonne donnée.
    
    Args:
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne pour laquelle calculer la moyenne.
        
    Returns:
        float: La moyenne empirique de la colonne.
    """
    # Exclure les valeurs manquantes
    values = data[column].dropna()

    # Calculer la moyenne empirique
    mean_value = data[column].sum()/len(values)

    return mean_value

def empirical_variance(data, column):
    """
    Calcule la variance empirique pour une colonne donnée, sans utiliser var().
    
    Args:
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne pour laquelle calculer la variance.
        
    Returns:
        float: La variance empirique de la colonne.
    """
    # Exclure les valeurs manquantes
    values = data[column].dropna()
    
    # Calculer la moyenne empirique
    mean_value = empirical_mean(data, column)
    
    # Calculer la somme des carrés des différences par rapport à la moyenne
    squared_diffs = (values - mean_value) ** 2
    variance_value = squared_diffs.sum() / len(values) 
    
    return variance_value

def empirical_skewness(data, column):
    """
    Calcule l'asymétrie empirique pour une colonne donnée, sans utiliser skew().
    
    Args:
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne pour laquelle calculer l'asymétrie.
        
    Returns:
        float: L'asymétrie empirique de la colonne.
    """
    # Exclure les valeurs manquantes
    values = data[column].dropna()
    
    # Calculer la moyenne empirique
    mean_value = empirical_mean(data, column)
    
    # Calculer la variance empirique
    variance_value = empirical_variance(data, column)
    
    # Calculer la somme des termes de la forme (x - mean)^3 / (n * variance^3)
    empirical_skewness = ((values - mean_value)**3).sum() / (len(values) * variance_value ** (3/2))
    
    return empirical_skewness

def empirical_kurtosis(data, column):
    """
    Calcule la kurtosis empirique pour une colonne donnée, sans utiliser kurtosis().
    
    Args:
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne pour laquelle calculer la kurtosis.

    Returns:
        float: La kurtosis empirique de la colonne.
    """
    # Exclure les valeurs manquantes
    values = data[column].dropna()
    
    # Calculer la moyenne empirique
    mean_value = empirical_mean(data, column)
    
    # Calculer la variance empirique
    variance_value = empirical_variance(data, column)
    
    # Calculer la somme des termes de la forme (x - mean)^4 / (n * variance^4)
    empirical_kurtosis = ((values - mean_value)**4).sum() / (len(values) * variance_value ** 2)
    
    return empirical_kurtosis