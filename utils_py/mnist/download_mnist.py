import os
import gzip
import urllib.request
import logging

DATASET_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_DIR = "data/mnist"


def download_mnist(data_dir):
    """
    Download the MNIST dataset.

    This function performs the following steps:
    1. Create the necessary directories if they do not exist.
    2. Check if the dataset is already downloaded.
    3. Download the dataset files if they are not already present.

    :return: None
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(MNIST_DIR):
        os.makedirs(MNIST_DIR)

    if all(os.path.exists(f"{MNIST_DIR}/{file}.txt") for file in ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]):
        print("MNIST dataset already downloaded.")
        return

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    headers = {'User-Agent': 'Mozilla/5.0'}

    for name, file_url in files.items():
        print(f"Downloading {name}...")
        req = urllib.request.Request(DATASET_URL + file_url, headers=headers)
        with urllib.request.urlopen(req) as response, open(f"{MNIST_DIR}/{file_url}", 'wb') as out_file:
            out_file.write(response.read())
        print("Done.")


def extract_mnist():
    """
    Extract the MNIST dataset files.

    This function performs the following steps:
    1. Set up logging.
    2. Extract `.gz` files to `.txt` files.
    3. Delete the original `.gz` files after extraction.

    :return: None
    """
    logging.basicConfig(level=logging.INFO)
    for filename in os.listdir(MNIST_DIR):
        if filename.endswith(".gz"):
            full_path = os.path.join(MNIST_DIR, filename)
            txt_filename = full_path[:-3] + ".txt"
            try:
                logging.info(f"Extracting {filename} to {txt_filename}...")
                with gzip.open(full_path, 'rb') as f_in, open(txt_filename, 'wb') as f_out:
                    f_out.write(f_in.read())
                logging.info(f"Done extracting {filename}.")
                os.remove(full_path)  # Delete the original .gz file
            except Exception as e:
                logging.error(f"Failed to extract {filename}: {e}")
