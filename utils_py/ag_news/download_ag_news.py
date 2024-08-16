import os
import requests

DATASET_NAME = "ag_news"


def download_ag_news(data_dir):
    """
    Load the AG_NEWS dataset and save it in the specified directory.

    Parameters:
    data_dir (str): The base directory where the data should be stored.
    dataset_name (str): The name of the dataset directory.

    Returns:
    None
    """
    base_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
    files = ["train.csv", "test.csv"]

    # Ensure base data directory exists
    if not os.path.exists(data_dir):
        print("Creating data directory...")
        os.makedirs(data_dir, exist_ok=True)

    # Ensure the dataset-specific directory exists
    dataset_dir = os.path.join(data_dir, DATASET_NAME)
    if not os.path.exists(dataset_dir):
        print(f"Creating {DATASET_NAME} dataset directory...")
        os.makedirs(dataset_dir, exist_ok=True)

    print(f"Downloading {DATASET_NAME} dataset...")
    for file in files:
        url = base_url + file
        response = requests.get(url)
        file_path = os.path.join(dataset_dir, file)
        with open(file_path, 'wb') as f:
            print(f"Downloading {file} to {file_path}...")
            f.write(response.content)

    print(f"Download complete.")
