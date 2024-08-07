from utils.mnist.download_mnist import download_mnist, extract_mnist, MNIST_DIR
from utils.mnist.load_data import load_data
from utils.mnist.normalize_data import normalize_data
from utils.mnist.update_files import update_files
from utils.ag_news.download_ag_news import download_ag_news
from utils.ag_news.extract_ag_news import extract_ag_news

import os

DATA_DIR = "data"

if __name__ == "__main__":
    print("Downloading and processing MNIST dataset...")
    if not all(os.path.exists(f"{MNIST_DIR}/{file}.txt") for file in
           ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]):
        # Download the MNIST dataset if not already downloaded
        download_mnist(DATA_DIR)

        # Extract the downloaded dataset files
        extract_mnist()

        # Load the training images and labels from the extracted files
        train_images, _ = load_data(image_file=f"{MNIST_DIR}/train-images-idx3-ubyte.txt",
                                    label_file=f"{MNIST_DIR}/train-labels-idx1-ubyte.txt")

        # Normalize the training images to a standard scale
        train_images = normalize_data(train_images)

        # Update the training image files with the normalized data
        update_files([f"{MNIST_DIR}/train-images-idx3-ubyte.txt"], train_images, update_type='data')

        # Load the test images and labels from the extracted files
        test_images, _ = load_data(image_file=f"{MNIST_DIR}/t10k-images-idx3-ubyte.txt",
                                   label_file=f"{MNIST_DIR}/t10k-labels-idx1-ubyte.txt")

        # Normalize the test images to a standard scale
        test_images = normalize_data(test_images)

        # Update the test image files with the normalized data
        update_files([f"{MNIST_DIR}/t10k-images-idx3-ubyte.txt"], test_images, update_type='data')

        # Print a success message indicating the completion of all steps
        print("MNIST data has been successfully downloaded, extracted, normalized, and updated.\n")
    else:
        print("MNIST dataset already downloaded and processed")

    print("Downloading and processing AG_NEWS dataset...")
    if all(os.path.exists(f"{DATA_DIR}/ag_news/{file}") for file in ["train.csv", "test.csv"]):
        # Download the AG_NEWS dataset if not already downloaded
        download_ag_news(DATA_DIR)

        # Extract the downloaded dataset files
        extract_ag_news(DATA_DIR)

        # Print a success message indicating the completion of all steps
        print("AG_NEWS data has been successfully downloaded, extracted, and cleaned.")
