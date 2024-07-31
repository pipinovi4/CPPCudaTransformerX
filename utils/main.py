from download_mnist import download, extract_mnist, DATASET_DIR

if __name__ == "__main__":
    download()
    train_images = extract_mnist(f"{DATASET_DIR}/train-labels-idx1-ubyte.gz")
    train_labels = extract_mnist(f"{DATASET_DIR}/train-labels-idx1-ubyte.gz")
    test_images = extract_mnist(f"{DATASET_DIR}/t10k-images-idx3-ubyte.gz")
    test_labels = extract_mnist(f"{DATASET_DIR}/t10k-labels-idx1-ubyte.gz")
    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)
