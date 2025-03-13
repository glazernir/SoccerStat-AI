import pandas as pd
from sklearn.model_selection import train_test_split


# Function to analyze the distribution of values in a specific column of a dataset
# Use this when interested in selecting a test set that contains only a certain feature/group of features
def analyze_stats(file_path, column_name):
    df = pd.read_csv(file_path)  # Load the dataset
    df[column_name] = df[column_name].dropna()  # Remove NaN values from the specified column

    column_length = len(df[column_name])  # Total non-null entries in the column
    threshold = 0.2 * column_length  # Compute 20% threshold for filtering

    value_counts = df[column_name].value_counts()  # Count occurrences of unique values in the column

    # Display results
    print("Value counts for the column:")
    print(value_counts)
    print(f"Column length: {column_length}")
    print(f"20% threshold: {threshold}")


# Function to create a test set consisting of players from a specific league
# Use this when the test set should be from a specific competition
def create_test_set(league):
    df = pd.read_csv("datasets/appearances.csv")  # Load the dataset containing player appearances
    filtered_df = df[df['competition_id'] == league]  # Filter players belonging to the specified league
    player_ids = filtered_df['player_id'].unique()  # Extract unique player IDs from the league
    return player_ids  # Return the list of player IDs for the test set


# Function to split data into train and test sets
# 'C' -> Split by competition (players from a specific league are used as the test set)
# 'R' -> Random split of the dataset
def split_to_train_test(results, char, player_ids):
    if char == 'C':  # If splitting by competition
        test_data = results[results['player_id'].isin(player_ids)]  # Test set: players in the specified league
        train_data = results[~results['player_id'].isin(player_ids)]  # Train set: all other players
        return test_data, train_data
    elif char == 'R':  # If splitting randomly
        train_data, test_data = train_test_split(results, test_size=0.2, random_state=42)  # 80-20 split
        return test_data, train_data


# Function to perform train-test split based on chosen split strategy
def run_train_test_create(df, split_shape):
    # The analyze_stats function can be used to check the most common leagues for selecting a test set
    # Example usage:
    # file_path = "datasets/appearances.csv"
    # column_name = "competition_id"
    # analyze_stats(file_path, column_name)

    # If split is based on competition ('C'), select players from the 'FR1' league for the test set
    if split_shape == 'C':
        player_ids = create_test_set('FR1')
        return split_to_train_test(df, split_shape, player_ids)
    else:  # If random split ('R')
        return split_to_train_test(df, split_shape, None)
