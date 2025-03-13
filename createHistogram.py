import pandas as pd
import matplotlib.pyplot as plt


def create_histograms(file_path):
    """
    Reads a CSV file and generates histograms for all numeric columns except 'player_id'.

    Parameters:
        file_path (str): Path to the CSV file.
    """
    data = pd.read_csv(file_path)

    # Select numeric columns, excluding 'player_id' if present
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = numeric_columns.drop('player_id', errors='ignore')

    # Create subplots based on the number of numeric columns
    fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(8, len(numeric_columns) * 4))

    # Ensure axes is iterable when there's only one column
    if len(numeric_columns) == 1:
        axes = [axes]

    # Generate histograms for each numeric column
    for ax, column in zip(axes, numeric_columns):
        ax.hist(data[column], bins=20, edgecolor='black')
        ax.set_title(f'Histogram of {column}')
        ax.set_xlabel('Value Ranges')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.show()


def run_create_histograms(file_path):
    """
    Calls the create_histograms function with the specified file path.

    Parameters:
        file_path (str): Path to the CSV file.
    """
    create_histograms(file_path)
