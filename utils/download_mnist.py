import os
import gzip
import numpy as np
import urllib.request

DATA_DIR = "../data"
DATASET_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
DATASET_DIR = "../data/mnist"


def download():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    if os.path.exists(f"{DATASET_DIR}/train-images-idx3-ubyte.gz"):
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
        with urllib.request.urlopen(req) as response, open(f"{DATASET_DIR}/{file_url}", 'wb') as out_file:
            out_file.write(response.read())
        print("Done.")


def extract_mnist(filename):
    with gzip.open(filename, "rb") as f:
        if "images" in filename:
            data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        else:
            data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data
