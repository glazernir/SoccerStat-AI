import pandas as pd
import matplotlib.pyplot as plt

from performance_vectors import weighted_data

def create_histogram(file_path):
    data = pd.read_csv(file_path)  # Replace "your_data.csv" with your actual file path

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

if __name__ == '__main__':
    #create_histogram(r'datasets/vector_appearances.csv')
    create_histogram(r'datasets/weighted_vector_appearances.csv')
