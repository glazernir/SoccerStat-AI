import pandas as pd

#Use When interested in choosing a test set that contains only a certain feature / group of features

def printStats(file_path, cloumn_name):
    df = pd.read_csv(file_path)
    df[column_name] = df[column_name].dropna()
    column_length = len(df[column_name])
    threshold = 0.2 * column_length
    value_counts = df[column_name].value_counts()
    print("Value counts for the column:")
    print(value_counts)
    print(f"Column length: {column_length}")
    print(f"20% threshold: {threshold}")


if __name__ == '__main__':

    file_path = "datasets/appearances.csv"
    column_name = "competition_id"
    printStats(file_path, column_name)
