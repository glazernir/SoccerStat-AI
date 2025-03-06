import numpy as np
import pandas as pd
#from train import trainDataset
from numpy.random import exponential
from sklearn.model_selection import train_test_split
from trainTest import train_and_test_autoencoder
decay_rate = 0.5

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def remove_insufficient_data_players(df):
    player_groups = df.groupby('player_id').agg(
        first_appearance=('date', 'min'),
        last_appearance=('date', 'max'),
        total_minutes=('minutes_played', 'sum')
    )

    player_groups['time_span_days'] = (player_groups['last_appearance'] - player_groups['first_appearance']).dt.days
    # remove players with less than a year of date, players with less than 30 minutes played 
    filtered_players = player_groups[
        (player_groups['time_span_days'] >= 365) & 
        (player_groups['total_minutes'] > 30)
    ]
    
    df_filtered =  df[df['player_id'].isin(filtered_players.index)]
    one_year_ago = pd.Timestamp.now() - pd.DateOffset(days=365)
    df_recent = df_filtered[df_filtered['date'] >= one_year_ago]
    
    # remove players who don't have data in the last year
    players_with_recent_data = df_recent['player_id'].unique()
    
    return df_filtered[df_filtered['player_id'].isin(players_with_recent_data)]

def calculate_features(player_df, method='weighted', decay_rate=0.5, time_factor=30):
    player_df['date'] = pd.to_datetime(player_df['date'], format='%d/%m/%Y')
    player_df = player_df[(player_df['date'].dt.year == 2012) & (player_df['date'].dt.month.isin([6,7,8]))]
    #player_df = player_df[(player_df['date'].dt.year.isin([2012,2013]))]

    player_performances = []
    i = 0
    for index, row in player_df.iterrows():
        print(i)
        i = i+1
        # make a copy of player_df and leave only rows that are closer to the current row than 90 days
        player_df_instance = player_df[(player_df['date'] <= row['date']) & (player_df['date'] >= row['date'] - pd.DateOffset(days=90))]
        total_minutes = player_df_instance['minutes_played'].sum()
        if total_minutes == 0:
            player_performance = pd.Series({
                                'normalized_goals': 0,
                                'normalized_assists': 0,
                                'normalized_yellow_cards': 0,
                                'normalized_red_cards': 0,
                                'total_time_in_minutes': 0
                                })

        else:
            if method == 'weighted':
                player_df_instance['calc_time'] = (row['date'] - player_df_instance['date']).dt.days
                player_df_instance['weights'] = np.exp(-decay_rate * player_df_instance['calc_time'] / time_factor)
                # player_df_instance['weights'] = player_df_instance['weights'] / player_df_instance['weights'].sum() # normalize
                #player_performance = pd.Series({col: (player_df_instance[col] * player_df_instance['weights']).sum() / total_minutes  for col in data_columns})

                player_performance = pd.Series({
                    'weighted_goals': (player_df_instance['goals'] * player_df_instance['weights']).sum() / total_minutes,
                    'weighted_assits': (player_df_instance['assists'] * player_df_instance['weights']).sum() / total_minutes,
                    'weighted_yellow_cards': (player_df_instance['yellow_cards'] * player_df_instance['weights']).sum() / total_minutes,
                    'weighted_red_cards': (player_df_instance['red_cards'] * player_df_instance['weights']).sum() / total_minutes,
                    'total_time': total_minutes / total_minutes
                })


            else:
                #player_performance = pd.Series({col: player_df_instance[col].sum() / total_minutes for col in data_columns})
                player_performance = pd.Series({'normalized_goals': player_df_instance['goals'].sum() / total_minutes,
                                                'normalized_assists': player_df_instance['assists'].sum() / total_minutes,
                                                'normalized_yellow_cards': player_df_instance['yellow_cards'].sum() / total_minutes,
                                                'normalized_red_cards': player_df_instance['red_cards'].sum() / total_minutes,
                                                'total_time_in_minutes': total_minutes / total_minutes})
            player_performance['minutes_played'] = total_minutes

        # add date, player name and ID
        # player_performance['date'] = row['date']
        # player_performance['player_club_id'] = row['player_club_id']
        player_performance['player_id'] = row['player_id']
        #player_performance['player_name'] = row['player_name']
        player_performances.append(player_performance)

    if not player_performances:  # Check if the list is empty
        return pd.DataFrame()
    return pd.concat(player_performances, axis=1).T


def create_performance_vectors(df):
    # Calculate for each timeframe
    performance_df = df.groupby('player_id').apply(lambda x: calculate_features(x))
    performance_df = pd.DataFrame(performance_df.values.tolist(), columns=performance_df.columns)
    return performance_df

def preprocessing(file_path):

    df = load_and_process_data(file_path)
    df = remove_insufficient_data_players(df)
    performance_df = create_performance_vectors(df)
    return performance_df

def run_performance_vectors(file_path):
    results = preprocessing(file_path)
    results.to_csv(r'datasets\prepared_data.csv')
    return results

