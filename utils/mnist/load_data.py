import numpy as np


def load_data(image_file, label_file):
    """
    Load image and label data from binary files.

    :param image_file: Path to the image file.
    :param label_file: Path to the label file.
    :return: Tuple of NumPy arrays (images, labels).
    """
    # Load images
    with open(image_file, 'rb') as f:
        magic_number = int(np.frombuffer(f.read(4), dtype=np.uint32).byteswap())
        num_images = int(np.frombuffer(f.read(4), dtype=np.uint32).byteswap())
        num_rows = int(np.frombuffer(f.read(4), dtype=np.uint32).byteswap())
        num_cols = int(np.frombuffer(f.read(4), dtype=np.uint32).byteswap())
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)

    # Load labels
    with open(label_file, 'rb') as f:
        magic_number = int(np.frombuffer(f.read(4), dtype=np.uint32).byteswap())
        num_labels = int(np.frombuffer(f.read(4), dtype=np.uint32).byteswap())
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels
