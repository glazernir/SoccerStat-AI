from coloring import run_coloring
from coloring import head_to_head_comparison
from chooseTestSet import run_train_test_create
import pandas as pd
from performance_vectors import run_performance_vectors
from trainTest import train_and_test_autoencoder


if __name__ == "__main__":
    # 1
    # load original players stats.
    file_path = 'datasets/appearances.csv'
    df = pd.read_csv(file_path)

    # 2
    # prepare data + weight data.
    prepared_data = run_performance_vectors(file_path)

    # 3
    # create Train - Test sets for PCA.
    # 'C' for split to train - test by competition,'R' for random split.
    test_data, train_data = run_train_test_create(prepared_data,'C')

    # 4
    # model training.
    print(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")
    model, test_loss = train_and_test_autoencoder(train_data, test_data, batch_size=32, encoding_dim=100, num_epochs=20,
                                                  learning_rate=0.003)
    print(f"Test Loss: {test_loss:.4f}")

    # 5
    # run PCA + display Coloring of PCA results.
    run_coloring()

    # 6
    # head-to-head comparison, by Player_id (hard Coded example):
    weightedData = pd.read_csv('datasets/Test.csv')
    playersData = pd.read_csv('datasets/players.csv')
    head_to_head_comparison(weightedData, playersData, 89200, 40680)
