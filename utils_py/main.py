from utils_py.mnist.download_mnist import download_mnist, extract_mnist, MNIST_DIR
from utils_py.mnist.load_data import load_data
from utils_py.mnist.normalize_data import normalize_data
from utils_py.mnist.update_files import update_files
from utils_py.ag_news.download_ag_news import download_ag_news
from utils_py.ag_news.extract_ag_news import extract_ag_news
from utils_py.wikitext.download_wikitext import download_wikitext
from utils_py.wikitext.extract_wikitext import extract_wikitext
from utils_py.vocab.download_vocab import download_vocab, train_vocab
from utils_py.vocab.extract_vocab import extract_vocab

import os

DATA_DIR = "data"

if __name__ == "__main__":
    print("Downloading and processing MNIST dataset...")
    if not all(os.path.exists(f"{MNIST_DIR}/{file}.txt") for file in
               ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte"]):
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
    if not all(os.path.exists(f"{DATA_DIR}/ag_news/{file}") for file in ["train.csv", "test.csv"]):
        # Download the AG_NEWS dataset if not already downloaded
        download_ag_news(DATA_DIR)

        # Extract the downloaded dataset files
        extract_ag_news(DATA_DIR)

        # Print a success message indicating the completion of all steps
        print("AG_NEWS data has been successfully downloaded, extracted, and cleaned.")

    if not all(os.path.exists(f"{DATA_DIR}/wikitext/{file}.txt") for file in ["train", "valid", "test"]):
        print("Downloading and processing WikiText-2 dataset...")
        # Download the WikiText-2 dataset if not already downloaded
        wikitext_dataset = download_wikitext()

        # Extract the downloaded dataset files
        extract_wikitext(wikitext_dataset, os.path.join(DATA_DIR, "wikitext"))

        # Print a success message indicating the completion of all steps
        print("WikiText-2 data has been successfully downloaded, extracted, and saved.")

    print("Downloading and processing SentencePiece vocabulary...")
    # Download and process the SentencePiece vocabulary
    vocab_file_path = download_vocab(os.path.join(DATA_DIR, "vocab"))
    train_vocab(vocab_file_path)
    extract_vocab(os.path.join(DATA_DIR, "vocab"))
    print("SentencePiece vocabulary has been successfully downloaded and processed.")
