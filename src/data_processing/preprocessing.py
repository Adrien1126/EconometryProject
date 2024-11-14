import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Charge les données depuis un fichier CSV et affiche les premières lignes.
    
    Args:
        filepath (str): Le chemin vers le fichier CSV.
    
    Returns:
        DataFrame: Les données chargées sous forme de DataFrame Pandas.
    """
    # Charger les données
    data = pd.read_csv(filepath)
    
    # Afficher les premières lignes pour vérifier le chargement
    print("Aperçu des données chargées :")
    print(data.head())
    
    return data


def check_missing_values(data):
    """
    Vérifie et affiche le nombre de valeurs manquantes pour chaque colonne d'un DataFrame.
    
    Args:
        data (DataFrame): Le DataFrame à vérifier.
    
    Returns:
        DataFrame: Un résumé du nombre de valeurs manquantes par colonne.
    """
    missing_values = data.isnull().sum()
    print("Valeurs manquantes par colonne :")
    print(missing_values[missing_values > 0])
    return missing_values

def price(data, log_price='logprice', return_column='price'):
    """
    Calcul des prix à partir des log-prices.
    
    Args:
        data (DataFrame): Le DataFrame contenant les données de logprice.
        log_price (str): Le nom de la colonne contenant les logprices.
        return_column (str): Le nom de la colonne qui contiendra les prix.
    
    Returns:
        DataFrame: Le DataFrame avec les prix calculés.
    """
    # Calcul des prix en utilisant numpy.exp()
    data[return_column] = np.exp(data[log_price])
    
    return data

def simple_return(data, price_column = 'Price', return_column = 'simple_return'):
    """
    Calcul des rendements simples à partir des prix.
    
    Args:
        data (DataFrame): Le DataFrame contenant les données de prix.
        price_column (str): Le nom de la colonne contenant les prix.
        return_column (str): Le nom de la colonne qui contiendra les rendements simples.
    
    Returns:
        DataFrame: Le DataFrame avec les rendements simples calculés.
    """
    # Calcul des rendements simples en calculant la différence entre les prix successifs
    data[return_column] = data[price_column].diff()/data[price_column].shift(1)
    
    return data

def compounded_return(data, log_price_column = 'logprice', return_column = 'coumpounded_return'):
    """
    Calcul des rendements compounded à partir des log-prices.
    
    Args:
        data (DataFrame): Le DataFrame contenant les données de logprice.
        log_price_column (str): Le nom de la colonne contenant les logprices.
        return_column (str): Le nom de la colonne qui contiendra les rendements compounded.
    
    Returns:
        DataFrame: Le DataFrame avec les rendements compounded calculés.
    """
    # Calcul des rendements compounded en utilisant la fonction cumprod() de numpy
    data[return_column] = data[log_price_column].diff()
    
    return data

def filter_outliers(data, column, std_dev_threshold=5):
    """
    Filtre les valeurs aberrantes dans une colonne donnée en supprimant
    les observations au-delà de ±k écarts-types par rapport à la moyenne.
    
    Args:
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne à filtrer.
        std_dev_threshold (float): Le nombre d'écarts-types à utiliser pour le filtrage (par défaut 5).
        
    Returns:
        DataFrame: Un DataFrame filtré sans les valeurs aberrantes.
    """
    # Calcul de la moyenne et de l'écart-type de la colonne
    mean = data[column].mean()
    std_dev = data[column].std()
    
    # Filtrer les valeurs en fonction du seuil en écart-type
    filtered_data = data[(data[column] >= mean - std_dev_threshold * std_dev) &
                         (data[column] <= mean + std_dev_threshold * std_dev)]
    
    return filtered_data

def calculate_daily_mean_returns(data, day_column='day', return_column='compounded_return'):
    """
    Calcule les moyennes journalières des rendements en utilisant un identifiant numérique de jour.
    
    Args:
        data (DataFrame): Le DataFrame contenant les données de returns.
        day_column (str): Le nom de la colonne contenant l'identifiant numérique du jour.
        return_column (str): Le nom de la colonne contenant les rendements.
        
    Returns:
        DataFrame: Un DataFrame avec les moyennes journalières des rendements.
    """
    # Calculer la moyenne journalière des rendements en groupant par 'day'
    daily_mean_returns = data.groupby(day_column)[return_column].mean().reset_index()
    
    # Renommer la colonne de return pour indiquer qu'il s'agit de moyennes journalières
    daily_mean_returns = daily_mean_returns.rename(columns={return_column: 'daily_mean_return'})
    
    return daily_mean_returns

def calculate_volatility(data, return_column='daily_mean_return'):
    """
    Calcule la volatilité journalière, mensuelle et annualisée des rendements.
    
    Args:
        data (DataFrame): Le DataFrame contenant les rendements journaliers.
        return_column (str): Le nom de la colonne contenant les rendements journaliers.
        
    Returns:
        dict: Un dictionnaire contenant les volatilités journalière, mensuelle et annualisée.
    """
    # Calcul de la volatilité journalière (écart-type des rendements journaliers)
    daily_volatility = data[return_column].std()

    # Calcul de la volatilité annualisée (en supposant environ 252 jours de trading par an)
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    # Calcul de la volatilité mensuelle
    monthly_volatility = daily_volatility / np.sqrt(12)
    
    
    return {
        'daily_volatility': daily_volatility,
        'monthly_volatility': monthly_volatility,
        'annualized_volatility': annualized_volatility
    }