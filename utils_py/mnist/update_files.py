import numpy as np


def update_files(file_paths, new_data, update_type='data'):
    """
    Update files with new data.

    This function performs the following steps:
    1. Validate that file_paths is a list and new_data is a NumPy array.
    2. Read data from each file, update the data with the new data, and save the updated data back to the file.
    3. Handle any potential errors.

    :param file_paths: List of file paths to be updated.
    :param new_data: NumPy array of new data to update the existing data.
    :param update_type: Type of data to update ('data' or 'labels').
    :return: None
    :raises ValueError: If file_paths is not a list, new_data is not a NumPy array, or update_type is invalid.
    """
    if not isinstance(file_paths, list):
        raise ValueError("file_paths must be a list.")
    if not isinstance(new_data, np.ndarray):
        raise ValueError("new_data must be a NumPy array.")
    if update_type not in ['data', 'labels']:
        raise ValueError("update_type must be either 'data' or 'labels'.")

    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as f:
                magic_number = int(np.frombuffer(f.read(4), dtype=np.uint32).byteswap())
                num_items = int(np.frombuffer(f.read(4), dtype=np.uint32).byteswap())
                if update_type == 'data':
                    num_rows = int(np.frombuffer(f.read(4), dtype=np.uint32).byteswap())
                    num_cols = int(np.frombuffer(f.read(4), dtype=np.uint32).byteswap())
                    data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)
                elif update_type == 'labels':
                    data = np.frombuffer(f.read(), dtype=np.uint8)

            # Make a writable copy of the data
            data = np.copy(data)

            if update_type == 'data':
                if new_data.shape != data.shape:
                    raise ValueError(
                        f"Shape mismatch: new_data shape {new_data.shape} does not match existing data shape {data.shape}")
                new_data = new_data.astype(data.dtype)  # Convert new_data to the same dtype as data
                np.copyto(data, new_data, where=new_data != 0)
            elif update_type == 'labels':
                if new_data.shape != data.shape:
                    raise ValueError(
                        f"Shape mismatch: new_data shape {new_data.shape} does not match existing data shape {data.shape}")
                data = new_data.astype(data.dtype)  # Ensure new_data is the same dtype as data

            with open(file_path, 'wb') as f:
                f.write(np.array([magic_number], dtype=np.uint32).byteswap().tobytes())
                f.write(np.array([num_items], dtype=np.uint32).byteswap().tobytes())
                if update_type == 'data':
                    f.write(np.array([num_rows], dtype=np.uint32).byteswap().tobytes())
                    f.write(np.array([num_cols], dtype=np.uint32).byteswap().tobytes())
                data.tofile(f)
        except Exception as e:
            raise ValueError(f"Failed to update file {file_path}: {e}")
