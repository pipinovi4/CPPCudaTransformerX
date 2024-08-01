from download_mnist import download, extract_mnist, DATASET_DIR
from load_data import load_data
from normalize_data import normalize_data
from update_files import update_files
import os

if __name__ == "__main__":
    if not all(os.path.exists(f"{DATASET_DIR}/{file}.txt") for file in ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]):
        # Download the MNIST dataset if not already downloaded
        download()

        # Extract the downloaded dataset files
        extract_mnist()

        # Load the training images and labels from the extracted files
        train_images, _ = load_data(image_file=f"{DATASET_DIR}/train-images-idx3-ubyte.txt",
                                    label_file=f"{DATASET_DIR}/train-labels-idx1-ubyte.txt")

        # Normalize the training images to a standard scale
        train_images = normalize_data(train_images)

        # Update the training image files with the normalized data
        update_files([f"{DATASET_DIR}/train-images-idx3-ubyte.txt"], train_images, update_type='data')

        # Load the test images and labels from the extracted files
        test_images, _ = load_data(image_file=f"{DATASET_DIR}/t10k-images-idx3-ubyte.txt",
                                   label_file=f"{DATASET_DIR}/t10k-labels-idx1-ubyte.txt")

        # Normalize the test images to a standard scale
        test_images = normalize_data(test_images)

        # Update the test image files with the normalized data
        update_files([f"{DATASET_DIR}/t10k-images-idx3-ubyte.txt"], test_images, update_type='data')

        # Print a success message indicating the completion of all steps
        print("MNIST data has been successfully downloaded, extracted, normalized, and updated.")
    else:
        test_images, _ = load_data(image_file=f"{DATASET_DIR}/t10k-images-idx3-ubyte.txt",
                                   label_file=f"{DATASET_DIR}/t10k-labels-idx1-ubyte.txt")
        print("MNIST dataset already downloaded and processed")
