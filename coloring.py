import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def coloringByParam(path,param,coloringShape):
    import pandas as pd
    from sklearn.decomposition import PCA

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

    #if coloringShape is 0 than categorical, else continuous
    if coloringShape == 0:
        continuousColoring(result,param)
        return

    plt.figure(figsize=(14, 10))
    sub_positions = result[param].unique()
    colors = plt.cm.tab10(range(len(sub_positions)))
    for sub_position, color in zip(sub_positions, colors):
        subset = result[result[param] == sub_position]
        plt.scatter(subset['PC1'], subset['PC2'], label=sub_position, color=color)
    plt.legend(title=param)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Scatter Plot of Points Colored by " + param)
    plt.show()

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

if __name__ == '__main__':

    # categorical coloring:

    # prepare_marketVaule()
    # coloringByParam('categorical_market_value.csv','categorical_market_value',1)

    # prepare_age()
    # coloringByParam("categorical_age.csv", 'categorical_age',1)

    # coloringByParam('datasets/players.csv', 'position',1)

    # prepare_League()
    # coloringByParam('categorical_League.csv','current_club_domestic_competition_id',1)

    # prepare_country()
    # coloringByParam('categorical_country.csv','country_of_birth',1)

    #coloringByParam('datasets/players.csv','foot',1)

    #continuous coloring:
    #coloringByParam('datasets/players.csv','market_value_in_eur',0)

    prepare_definiteTime_stats('datasets/players.csv',2020)
    prepare_age(2020,str(2020) + "_stats.csv")
    coloringByParam('players_age.csv', 'age', 0)

