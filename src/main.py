from data_processing.preprocessing import load_data, check_missing_values, calculate_log_returns
from utils.plotting import plot_log_prices

def main():
    # Chemin du fichier CSV
    filepath = 'data/raw/out12.csv'
    
    # Charger les données
    data = load_data(filepath)
    
    # Vérifier les valeurs manquantes
    check_missing_values(data)
    
    # Calculer les log-returns
    data = calculate_log_returns(data, price_column='logprice', return_column='log_return')
    
    # Afficher un aperçu des données avec log-returns
    print(data[['day', 'tick', 'logprice', 'log_return']].head())

    plot_log_prices(data, return_column='logprice')


if __name__ == "__main__":
    main()
