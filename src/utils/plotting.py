import matplotlib.pyplot as plt

def plot_price(data, return_column = 'logprice'):
    """
    Trace les logprice en fonction du temps.
    
    Args:
        data (DataFrame): Le DataFrame contenant les données de logprice.
        return_column (str): Le nom de la colonne contenant les logprices.
        
    Returns:
        None: Affiche le graphique.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data[return_column], color='black')  # Tracer en noir
    plt.title(f'Values of the {return_column} variable against time', color='black')
    plt.xlabel('Index', color='black')
    plt.ylabel(return_column, color='black')
    
    # Ajuster les couleurs des axes en noir
    plt.tick_params(axis='x', colors='black')
    plt.tick_params(axis='y', colors='black')
    
    # Afficher le graphique
    plt.show()

def plot_return(data, return_column):
    """
    Trace les rendements en fonction du temps.
    
    Args:
        data (DataFrame): Le DataFrame contenant les données de rendements.
        return_column (str): Le nom de la colonne contenant les rendements.
        
    Returns:
        None: Affiche le graphique.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data[return_column], color='black')  # Tracer en noir
    plt.title(f'Values of the {return_column} variable against time', color='black')
    plt.xlabel('Index', color='black')
    plt.ylabel(return_column, color='black')
    
    # Ajuster les couleurs des axes en noir
    plt.tick_params(axis='x', colors='black')
    plt.tick_params(axis='y', colors='black')
    
    # Afficher le graphique
    plt.show()

def plot_boxplot(data, column='log_return'):
    """
    Affiche un box plot pour la colonne spécifiée d'un DataFrame.
    
    Args:
        data (DataFrame): Le DataFrame contenant les données.
        column (str): Le nom de la colonne pour laquelle le box plot sera tracé.
        
    Returns:
        None: Affiche le graphique.
    """
    plt.figure(figsize=(12, 6))
    plt.boxplot(data[column].dropna(), vert=False)
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.grid(True)
    plt.show()