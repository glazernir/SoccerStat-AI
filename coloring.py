import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from prettytable import PrettyTable


# Function to compare two players based on weighted statistics and player information
def head_to_head_comparison(weighted_data, players_data, player1_id, player2_id):
    # Extract the relevant player data
    player1_weighted = weighted_data[weighted_data['player_id'] == player1_id].iloc[0]
    player2_weighted = weighted_data[weighted_data['player_id'] == player2_id].iloc[0]

    # Drop unnecessary columns (image_url, url) from player data
    player1_players = players_data[players_data['player_id'] == player1_id].drop(columns=['image_url', 'url'], errors='ignore').iloc[0]
    player2_players = players_data[players_data['player_id'] == player2_id].drop(columns=['image_url', 'url'], errors='ignore').iloc[0]

    # Create a table to display statistics comparison
    table = PrettyTable()
    table.field_names = ["Statistic", f"Player {player1_id}", f"Player {player2_id}"]

    # Add weighted stats comparison
    for column in weighted_data.columns.drop('player_id'):
        if column and not column.startswith('Unnamed'):  # Avoid empty and unnamed columns
            table.add_row([column, player1_weighted[column], player2_weighted[column]])

    # Add player personal details comparison
    for column in player1_players.index:
        table.add_row([column, player1_players[column], player2_players[column]])

    print(table)


# Function to plot dataset with continuous coloring based on a given parameter
def plot_dataset_with_continuous_coloring(data, param):
    def on_click(event):
        if event.inaxes:
            x_click, y_click = event.xdata, event.ydata
            distances = np.sqrt((data['PC1'] - x_click) ** 2 + (data['PC2'] - y_click) ** 2)
            closest_index = distances.idxmin()
            print(
                f"Clicked point: PC1={data.loc[closest_index, 'PC1']}, PC2={data.loc[closest_index, 'PC2']}, player_id={int(data.loc[closest_index, 'player_id'])}")

    # Scatter plot with color mapping based on the parameter
    plt.figure(figsize=(14, 10))
    sc = plt.scatter(data['PC1'], data['PC2'], c=data[param], cmap='Reds', s=100, alpha=0.8)
    cbar = plt.colorbar(sc)
    cbar.set_label(param)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Scatter Plot of Points Colored by {param}")
    plt.tight_layout()

    # Enable interactive click event
    fig = plt.gcf()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()


# Function to plot dataset with categorical coloring based on a given parameter
def plot_dataset_with_categorical_coloring(data, param):
    def on_click(event):
        if event.inaxes:
            x_click, y_click = event.xdata, event.ydata
            distances = np.sqrt((data['PC1'] - x_click) ** 2 + (data['PC2'] - y_click) ** 2)
            closest_index = distances.idxmin()
            print(
                f"Clicked point: PC1={data.loc[closest_index, 'PC1']}, PC2={data.loc[closest_index, 'PC2']}, player_id={int(data.loc[closest_index, 'player_id'])}")

    plt.figure(figsize=(14, 10))
    param_values = data[param].unique()
    colors = plt.cm.tab10(range(len(param_values)))

    # Scatter plot for each categorical group
    for sub_position, color in zip(param_values, colors):
        subset = data[data[param] == sub_position]
        plt.scatter(subset['PC1'], subset['PC2'], label=sub_position, color=color, picker=True)

    plt.legend(title=param)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Scatter Plot of Points Colored by {param}")

    fig = plt.gcf()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()


# Function to process and plot data with continuous or categorical coloring
def coloring_by_param(path, param, coloring_shape):
    # Load player data and extract the relevant parameter
    players = pd.read_csv(path)
    w_players = players[["player_id", param]]

    # Load PCA-transformed dataset
    df = pd.read_csv(r'expanded_data_train.csv')
    player_numbers = df['player_id']
    df_toReduce = df.drop(columns=["player_id"])

    # Perform PCA reduction to 2 components
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_toReduce.iloc[:, :-1])
    reduced_data = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
    reduced_data["player_id"] = player_numbers.values

    # Merge with the selected parameter
    result = reduced_data.merge(w_players, on="player_id", how="left")
    result.to_csv('pca_results.csv', index=False)

    # Choose the coloring type (continuous or categorical)
    if coloring_shape == 0:
        plot_dataset_with_continuous_coloring(result, param)
    else:
        plot_dataset_with_categorical_coloring(result, param)


# Function to preprocess dataset and extract common leagues
def prepare_league():
    players = pd.read_csv("datasets/players.csv")
    most_common_leagues = players['current_club_domestic_competition_id'].value_counts().head(4).index
    players = players[players['current_club_domestic_competition_id'].isin(most_common_leagues)]
    players.to_csv("categorical_League.csv")


# Function to preprocess dataset and extract most common birth countries
def prepare_country():
    players = pd.read_csv("datasets/players.csv")
    most_common_countries = players['country_of_birth'].value_counts().head(10).index
    players = players[players['country_of_birth'].isin(most_common_countries)]
    players.to_csv("categorical_country.csv")


# Function to calculate player age based on a target year
def prepare_age(year, path):
    players = pd.read_csv(path)
    players['date_of_birth'] = pd.to_datetime(players['date_of_birth'])
    players.loc[:, 'age'] = year - players['date_of_birth'].dt.year
    players.to_csv("players_age.csv", index=False)


# Function to filter player statistics for a specific year
def prepare_definite_time_stats(path, requested_year):
    original_df = pd.read_csv(r'datasets/appearances.csv')
    original_df['date'] = pd.to_datetime(original_df['date'])
    df_reqYear = original_df[original_df['date'].dt.year == requested_year]
    players_in_requested_year = df_reqYear['player_id'].unique()

    # Load player information
    players = pd.read_csv('datasets/players.csv')
    w_players = players[["player_id", "date_of_birth"]]

    # Load and apply PCA to dataset
    df = pd.read_csv(r'expanded_data_train.csv')
    player_numbers = df['player_id']
    df_toReduce = df.drop(columns=["player_id"])
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_toReduce.iloc[:, :-1])
    reduced_data = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
    reduced_data["player_id"] = player_numbers.values

    # Filter players from the requested year and save results
    df_players_filtered = reduced_data[reduced_data['player_id'].isin(players_in_requested_year)]
    result = df_players_filtered.merge(w_players, on="player_id", how="left")
    result.to_csv(f"{requested_year}_stats.csv", index=False)


# Function to preprocess data for coloring
def preprocessing():
    prepare_league()
    prepare_country()
    year = 2014
    prepare_definite_time_stats('datasets/players.csv', year)
    prepare_age(year, f"{year}_stats.csv")


# Function to run all coloring tasks
def run_coloring():
    preprocessing()

    # Categorical coloring
    coloring_by_param('datasets/players.csv', 'position', 1)
    coloring_by_param('categorical_League.csv', 'current_club_domestic_competition_id', 1)
    coloring_by_param('categorical_country.csv', 'country_of_birth', 1)
    coloring_by_param('datasets/players.csv', 'foot', 1)

    # Continuous coloring
    coloring_by_param('datasets/players.csv', 'market_value_in_eur', 0)
    coloring_by_param('datasets/players.csv', 'height_in_cm', 0)
    coloring_by_param('players_age.csv', 'age', 0)
