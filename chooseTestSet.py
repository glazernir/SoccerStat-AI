import pandas as pd
from sklearn.model_selection import train_test_split

#Use When interested in choosing a test set that contains only a certain feature / group of features
def analyzeStats(file_path, column_name):
    df = pd.read_csv(file_path)
    df[column_name] = df[column_name].dropna()
    column_length = len(df[column_name])
    threshold = 0.2 * column_length
    value_counts = df[column_name].value_counts()
    print("Value counts for the column:")
    print(value_counts)
    print(f"Column length: {column_length}")
    print(f"20% threshold: {threshold}")

#Use when the test set is a specific league
def createTestSet(league):
    # extract the players from specific league for test set
    df = pd.read_csv("datasets/appearances.csv")
    filtered_df = df[df['competition_id'] == league]
    player_ids = filtered_df['player_id'].unique()
    return player_ids

def splitToTrainTest(results,char,player_ids):
    # if we want to split by competition
    if char == 'C':
        test_data = results[results['player_id'].isin(player_ids)]
        train_data = results[~results['player_id'].isin(player_ids)]
        return test_data, train_data
    # if we want to split randomly
    elif char == 'R':
        train_data, test_data = train_test_split(results, test_size=0.2, random_state=42)
        return test_data, train_data

def run_trainTestCreate(df,splitShape):

    # You can check which are the most common leagues among the dataset for Test set with the help of the function analyzeStats.
    # For example:
    # file_path = "datasets/appearances.csv"
    # column_name = "competition_id"
    # analyzeStats(file_path, column_name)

    # 'C' for split to train - test by competition,'R' for random split.
    if splitShape == 'C':
        player_ids = createTestSet('FR1')
        return splitToTrainTest(df,splitShape,player_ids)
    else:
        return splitToTrainTest(df,splitShape,None)
