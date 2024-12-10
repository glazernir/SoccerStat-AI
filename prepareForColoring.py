import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



def coloringByParam(path,param):
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

def prepare_marketVaule():
    players = pd.read_csv("datasets/players.csv")
    players["categorical_market_value"],ranges = pd.qcut(players['market_value_in_eur'], q=4, labels=['A', 'B', 'C', 'D'],retbins=True)
    players.to_csv("categorical_market_value.csv", index=False)
    return players

def setMaxAge(players):
    return players[players['last_season'] < 2017]

def prepare_age():
    players = pd.read_csv("datasets/players.csv")
    target_year = 2024
    players['date_of_birth'] = pd.to_datetime(players['date_of_birth'], dayfirst=False)
    players.loc[:, 'age'] = target_year - players['date_of_birth'].dt.year
    players.loc[:, 'categorical_age'], bins = pd.qcut(players['age'], q=4,
                                                               labels=['1-10', '10-15', '15-20', '20-42'], retbins=True)
    players.to_csv("categorical_age.csv", index=False)

if __name__ == '__main__':
    # prepare_marketVaule()
    # coloringByParam('categorical_market_value.csv','categorical_market_value')
    prepare_age()
    coloringByParam("categorical_age.csv", 'categorical_age')
    #coloringByParam('datasets/players.csv', 'position')


