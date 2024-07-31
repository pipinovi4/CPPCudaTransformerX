import gzip
import numpy as np
import urllib.request


def download(url, filename):
    base_url = "https://yann.lecun.com/exdb/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    for name, url in files.items():
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(base_url + url, filename + url)
        print("Done.")


def extract_mnist(filename):
    with gzip.open(filename, "rb") as f:
        if "images" in filename:
            data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        else:
            data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data
