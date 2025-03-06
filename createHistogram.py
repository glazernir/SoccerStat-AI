import pandas as pd
import matplotlib.pyplot as plt


def create_histogram(file_path):
    data = pd.read_csv(file_path)
    numeric_columns = data.drop(columns=['player_id'], errors='ignore').select_dtypes(
        include=['float64', 'int64']).columns

    fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(8, len(numeric_columns) * 4))
    for i, column in enumerate(numeric_columns):
        ax = axes[i] if len(numeric_columns) > 1 else axes
        ax.hist(data[column], bins=20, edgecolor='black')
        ax.set_title(f'Histogram of {column}')
        ax.set_xlabel('Value Ranges')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.show()

def run_createHistogram(file_path):
    create_histogram(r'datasets/prepared_data.csv')

