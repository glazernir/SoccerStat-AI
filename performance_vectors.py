import pandas as pd

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
    
    # remove players who dont have data in the last year
    players_with_recent_data = df_recent['player_id'].unique()
    
    return df_filtered[df_filtered['player_id'].isin(players_with_recent_data)]

def calculate_sums(player_df):
    total_minutes = player_df['minutes_played'].sum()
    if total_minutes == 0:
        return pd.Series({
            'normalized_goals': 0,
            'normalized_assists': 0,
            'normalized_yellow_cards': 0,
            'normalized_red_cards': 0,
            'total_time_in_minutes': 0
        })
    return pd.Series({
        'normalized_goals': player_df['goals'].sum() / total_minutes,
        'normalized_assists': player_df['assists'].sum() / total_minutes,
        'normalized_yellow_cards': player_df['yellow_cards'].sum() / total_minutes,
        'normalized_red_cards': player_df['red_cards'].sum() / total_minutes,
        'total_time_in_minutes': total_minutes
    })


def calc_time_limit_ago(time_limit_ago, df):
    df_last_year = df[df['date'] >= time_limit_ago]
    return df_last_year.groupby('player_id').apply(calculate_sums)

def calc_by_timefranes(df):
    timeframes = {
        'one_month_ago': pd.Timestamp.now() - pd.DateOffset(days=30),
        'three_months_ago': pd.Timestamp.now() - pd.DateOffset(days=90),
        'six_months_ago': pd.Timestamp.now() - pd.DateOffset(days=182),
        'one_year_ago': pd.Timestamp.now() - pd.DateOffset(days=365)
    }
    
    results = pd.DataFrame()

    for label, time_limit in timeframes.items():
        # Calculate for each timeframe
        result = calc_time_limit_ago(time_limit, df)
        
        # Rename columns to indicate the time period
        result.columns = [f'{col}_{label}' for col in result.columns]
        
        # Concatenate the results by player_id
        if results.empty:
            results = result
        else:
            results = results.join(result, how='outer')

    return results


if __name__ == "__main__":
    file_path = r'datasets\appearances.csv'
    df = load_and_process_data(file_path)
    df = remove_insufficient_data_players(df)
    results = calc_by_timefranes(df)
    
    print(results.head())

