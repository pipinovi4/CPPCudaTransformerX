import torch


def normalize_data(data):
    """
    Normalize MNIST data using Z-score normalization with PyTorch.

    This function performs the following steps:
    1. Replace NaN values with 0.
    2. Clip extreme values to a reasonable range (0 to 1).
    3. Normalize the data using Z-score normalization.

    :param data: NumPy array of MNIST data.
    :return: Normalized NumPy array of MNIST data.
    """
    # Convert data to PyTorch tensor
    data = torch.tensor(data, dtype=torch.float32)

    # Replace NaN values with 0
    data = torch.nan_to_num(data, nan=0.0)

    # Clip extreme values to a reasonable range (e.g., 0 to 1)
    data = torch.clamp(data, 0, 1)

    # Normalize using Z-score normalization
    mean = torch.mean(data)
    std = torch.std(data)
    data = (data - mean) / std

    # Convert the normalized tensor back to a NumPy array
    return data.numpy()
