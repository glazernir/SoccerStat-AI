import numpy as np
import pandas as pd
# from train import trainDataset
from numpy.random import exponential
from sklearn.model_selection import train_test_split
from trainTest import train_and_test_autoencoder

decay_rate = 0.5  # Exponential decay rate for weighted feature calculation


# Load and preprocess data from a CSV file
def load_and_process_data(file_path):
    """
    Load data from a CSV file, convert the 'date' column to datetime format,
    and sort entries by date to ensure chronological order.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - A pandas DataFrame with sorted data.
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')  # Ensure chronological order
    return df


# Remove players who lack sufficient data for meaningful analysis
def remove_insufficient_data_players(df):
    """
    Filters out players with insufficient data based on:
    - A minimum total playing time of 30 minutes.
    - At least one year of recorded activity.
    - Having played within the last year.

    Parameters:
    - df: DataFrame containing player performance data.

    Returns:
    - A filtered DataFrame with only eligible players.
    """
    # Group data by player and compute key statistics
    player_groups = df.groupby('player_id').agg(
        first_appearance=('date', 'min'),
        last_appearance=('date', 'max'),
        total_minutes=('minutes_played', 'sum')
    )

    # Compute the active timespan for each player in days
    player_groups['time_span_days'] = (player_groups['last_appearance'] - player_groups['first_appearance']).dt.days

    # Filter players based on the defined criteria
    filtered_players = player_groups[
        (player_groups['time_span_days'] >= 365) &
        (player_groups['total_minutes'] > 30)
        ]

    # Keep only data from players who passed the filtering
    df_filtered = df[df['player_id'].isin(filtered_players.index)]

    # Determine players who have played within the last year
    one_year_ago = pd.Timestamp.now() - pd.DateOffset(days=365)
    df_recent = df_filtered[df_filtered['date'] >= one_year_ago]

    # Keep only players with at least one appearance in the last year
    players_with_recent_data = df_recent['player_id'].unique()
    return df_filtered[df_filtered['player_id'].isin(players_with_recent_data)]


# Compute performance metrics for players over time
def calculate_features(player_df, method='weighted', decay_rate=0.5, time_factor=30):
    """
    Computes normalized or weighted performance metrics for a player based on a 90-day history.

    Parameters:
    - player_df: DataFrame containing data for a single player.
    - method: 'weighted' (default) applies exponential decay, otherwise calculates raw averages.
    - decay_rate: The decay rate for weighting older data.
    - time_factor: The scaling factor for time decay.

    Returns:
    - A DataFrame with computed performance metrics for each appearance.
    """

    # can select only specific years for the data processing. for example:
    # player_df = player_df[(player_df['date'].dt.year.isin([2012, 2013]))]

    player_df['date'] = pd.to_datetime(player_df['date'], format='%d/%m/%Y')

    player_performances = []

    for index, row in player_df.iterrows():
        # Consider only past 90 days of performance before the current row's date
        player_df_instance = player_df[(player_df['date'] <= row['date']) &
                                       (player_df['date'] >= row['date'] - pd.DateOffset(days=90))]

        total_minutes = player_df_instance['minutes_played'].sum()

        if total_minutes == 0:
            # If no playing time, assign zeroed-out performance metrics
            player_performance = pd.Series({
                'normalized_goals': 0,
                'normalized_assists': 0,
                'normalized_yellow_cards': 0,
                'normalized_red_cards': 0,
                'total_time_in_minutes': 0
            })
        else:
            if method == 'weighted':
                # Compute time difference (in days) from the target row's date
                player_df_instance['calc_time'] = (row['date'] - player_df_instance['date']).dt.days

                # Compute exponentially decaying weights based on the time difference
                player_df_instance['weights'] = np.exp(-decay_rate * player_df_instance['calc_time'] / time_factor)

                # Compute weighted performance metrics
                player_performance = pd.Series({
                    'weighted_goals': (player_df_instance['goals'] * player_df_instance[
                        'weights']).sum() / total_minutes,
                    'weighted_assists': (player_df_instance['assists'] * player_df_instance[
                        'weights']).sum() / total_minutes,
                    'weighted_yellow_cards': (player_df_instance['yellow_cards'] * player_df_instance[
                        'weights']).sum() / total_minutes,
                    'weighted_red_cards': (player_df_instance['red_cards'] * player_df_instance[
                        'weights']).sum() / total_minutes,
                    'total_time': total_minutes / total_minutes  # This is always 1
                })
            else:
                # Compute raw averages (normalized metrics)
                player_performance = pd.Series({
                    'normalized_goals': player_df_instance['goals'].sum() / total_minutes,
                    'normalized_assists': player_df_instance['assists'].sum() / total_minutes,
                    'normalized_yellow_cards': player_df_instance['yellow_cards'].sum() / total_minutes,
                    'normalized_red_cards': player_df_instance['red_cards'].sum() / total_minutes,
                    'total_time_in_minutes': total_minutes / total_minutes  # Always 1
                })

            player_performance['minutes_played'] = total_minutes

        # Include player ID for reference
        player_performance['player_id'] = row['player_id']
        player_performances.append(player_performance)

    if not player_performances:  # Handle case where no valid data exists
        return pd.DataFrame()

    return pd.concat(player_performances, axis=1).T  # Combine results into a single DataFrame


# Generate player performance vectors for modeling
def create_performance_vectors(df):
    """
    Computes performance vectors for all players by applying feature extraction.

    Parameters:
    - df: DataFrame with game data for all players.

    Returns:
    - A DataFrame containing performance metrics for each player.
    """
    performance_df = df.groupby('player_id').apply(lambda x: calculate_features(x))
    performance_df = pd.DataFrame(performance_df.values.tolist(), columns=performance_df.columns)
    return performance_df


# Full preprocessing pipeline: loading data, filtering, and creating performance vectors
def preprocessing(file_path):
    """
    Executes the full data preprocessing pipeline.

    Parameters:
    - file_path: Path to the CSV file containing raw player data.

    Returns:
    - A DataFrame containing processed performance data.
    """
    df = load_and_process_data(file_path)
    df = remove_insufficient_data_players(df)
    performance_df = create_performance_vectors(df)
    return performance_df


# Save preprocessed data to a CSV file
def run_performance_vectors(file_path):
    """
    Runs the preprocessing pipeline and saves the output to a CSV file.

    Parameters:
    - file_path: Path to the CSV file containing raw player data.

    Returns:
    - A DataFrame containing processed performance data.
    """
    results = preprocessing(file_path)
    results.to_csv(r'datasets\prepared_data.csv', index=False)  # Save without row indices
    return results
