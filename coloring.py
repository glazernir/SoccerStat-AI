import pandas as pd
import matplotlib
from torch.utils.hipify.hipify_python import preprocessor

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np



def plot_dataset_with_continuous_coloring(data, param):
    def on_click(event):
        if event.inaxes:
            x_click, y_click = event.xdata, event.ydata
            distances = np.sqrt((data['PC1'] - x_click) ** 2 + (data['PC2'] - y_click) ** 2)
            closest_index = distances.idxmin()
            pc1 = data.loc[closest_index, 'PC1']
            pc2 = data.loc[closest_index, 'PC2']
            player_id = data.loc[closest_index, 'player_id']
            print(f"Clicked point: PC1={pc1}, PC2={pc2}, player_id={player_id}")

    plt.figure(figsize=(14, 10))
    sc = plt.scatter(
        data['PC1'], data['PC2'],
        c=data[param],
        cmap='Reds',
        s=100,
        alpha=0.8
    )
    cbar = plt.colorbar(sc)
    cbar.set_label(param)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Scatter Plot of Points Colored by " + param)
    plt.tight_layout()

    fig = plt.gcf()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()


def plot_dataset_with_categorical_coloring(data, param):
    def on_click(event):
        if event.inaxes:
            x_click, y_click = event.xdata, event.ydata
            distances = np.sqrt((data['PC1'] - x_click) ** 2 + (data['PC2'] - y_click) ** 2)
            closest_index = distances.idxmin()
            pc1 = data.loc[closest_index, 'PC1']
            pc2 = data.loc[closest_index, 'PC2']
            player_id = data.loc[closest_index, 'player_id']
            print(f"Clicked point: PC1={pc1}, PC2={pc2}, player_id={player_id}")

    plt.figure(figsize=(14, 10))
    param_values = data[param].unique()
    colors = plt.cm.tab10(range(len(param_values)))
    for sub_position, color in zip(param_values, colors):
        subset = data[data[param] == sub_position]
        plt.scatter(
            subset['PC1'],
            subset['PC2'],
            label=sub_position,
            color=color,
            picker=True
        )

    plt.legend(title=param)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Scatter Plot of Points Colored by " + param)

    fig = plt.gcf()  # Get current figure
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()


def coloringByParam(path,param,coloringShape):

    # Load the data
    players = pd.read_csv(path)
    w_players = players[["player_id", param]]
    df = pd.read_csv(r'expanded_data_train.csv')
    player_numbers = df['player_id']
    df_toReduce = df.drop(columns=["player_id"])
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_toReduce.iloc[:, :-1])
    reduced_data = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
    reduced_data["player_id"] = player_numbers.values
    result = reduced_data.merge(w_players, on="player_id", how="left")
    result.to_csv('pca_results.csv', index=False)

    #if coloringShape is 0 than continuous, else categorical
    if coloringShape == 0:
        plot_dataset_with_continuous_coloring(result, param)
        return

    plot_dataset_with_categorical_coloring(result,param)


def continuousColoring(dataSet,param):
    plt.figure(figsize=(14, 10))
    sc = plt.scatter(
        dataSet['PC1'], dataSet['PC2'],
        c=dataSet[param],
        cmap='Reds',
        s=100,
        alpha=0.8
    )
    cbar = plt.colorbar(sc)
    cbar.set_label(param)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Scatter Plot of Points Colored by " + param)
    plt.tight_layout()
    plt.show()


def prepare_marketVaule():
    players = pd.read_csv("datasets/players.csv")
    categorical_market_value, ranges = pd.qcut(
        players['market_value_in_eur'],
        q=4,
        retbins=True
    )
    labels = [f"{int(low)}-{int(high)}" for low, high in zip(ranges[:-1], ranges[1:])]
    players["categorical_market_value"] = pd.qcut(
        players['market_value_in_eur'],
        q=4,
        labels=labels
    )
    players.to_csv("categorical_market_value.csv", index=False)
    return players

def prepare_League():
    players = pd.read_csv("datasets/players.csv")
    most_common_leagues = players['current_club_domestic_competition_id'].value_counts().head(4).index
    players = players[players['current_club_domestic_competition_id'].isin(most_common_leagues)]
    players.to_csv("categorical_League.csv")

def prepare_country():
    players = pd.read_csv("datasets/players.csv")
    most_common_leagues = players['country_of_birth'].value_counts().head(10).index
    players = players[players['country_of_birth'].isin(most_common_leagues)]
    players.to_csv("categorical_country.csv")

def setMaxAge(players):
    return players[players['last_season'] < 2017]

def prepare_age(year,path):
    players = pd.read_csv(path)
    target_year = year
    players['date_of_birth'] = pd.to_datetime(players['date_of_birth'], dayfirst=False)
    players.loc[:, 'age'] = target_year - players['date_of_birth'].dt.year
    players.to_csv("players_age.csv", index=False)

def prepare_definiteTime_stats(path,requestedYear):
    original_df = pd.read_csv(r'datasets/appearances.csv')
    original_df['date'] = pd.to_datetime(original_df['date'], dayfirst=False)
    df_reqYear = original_df[original_df['date'].dt.year == requestedYear]
    players_in_2017 = df_reqYear['player_id'].unique()
    players = pd.read_csv('datasets/players.csv')
    w_players = players[["player_id","date_of_birth"]]
    df = pd.read_csv(r'expanded_data_train.csv')
    player_numbers = df['player_id']
    df_toReduce = df.drop(columns=["player_id"])
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_toReduce.iloc[:, :-1])
    reduced_data = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
    reduced_data["player_id"] = player_numbers.values
    df_players_in_2017 = reduced_data[reduced_data['player_id'].isin(players_in_2017)]
    result = df_players_in_2017.merge(w_players, on="player_id", how="left")
    result.to_csv(str(requestedYear) + '_stats.csv', index=False)


from prettytable import PrettyTable


def head_to_head_comparison(weightedData, playersData, player1_id, player2_id):
    player1_weighted = weightedData[weightedData['player_id'] == player1_id].iloc[0]
    player2_weighted = weightedData[weightedData['player_id'] == player2_id].iloc[0]
    player1_players = \
    playersData[playersData['player_id'] == player1_id].drop(columns=['image_url', 'url'], errors='ignore').iloc[0]
    player2_players = \
    playersData[playersData['player_id'] == player2_id].drop(columns=['image_url', 'url'], errors='ignore').iloc[0]

    table = PrettyTable()
    table.field_names = ["Statistic", f"Player {player1_id}", f"Player {player2_id}"]

    for column in weightedData.columns.drop('player_id'):
        table.add_row([column, player1_weighted[column], player2_weighted[column]])

    for column in player1_players.index:
        table.add_row([column, player1_players[column], player2_players[column]])

    print(table)


def Preprocessing():
    prepare_League()
    prepare_country()
    prepare_definiteTime_stats('datasets/players.csv', 2014)
    prepare_age(2014, str(2014) + "_stats.csv")


if __name__ == '__main__':

    Preprocessing()

    #categorical coloring:

    coloringByParam('datasets/players.csv', 'position',1)
    coloringByParam('categorical_League.csv','current_club_domestic_competition_id',1)
    coloringByParam('categorical_country.csv','country_of_birth',1)
    coloringByParam('datasets/players.csv','foot',1)

    #continuous coloring:

    coloringByParam('datasets/players.csv','market_value_in_eur',0)
    coloringByParam('datasets/players.csv','height_in_cm',0)
    coloringByParam('players_age.csv', 'age', 0)


    #head to head comparison, by Player_id:

    # weightedData = pd.read_csv('datasets/weighted_vector_appearances.csv')
    # playersData = pd.read_csv('datasets/players.csv')
    # head_to_head_comparison(weightedData,playersData, 542684,15452)



