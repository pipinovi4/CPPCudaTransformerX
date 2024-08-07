import logging
import os
import re
from utils.ag_news.download_ag_news import DATASET_NAME


def extract_ag_news(data_dir):
    """
    Extract the AG_NEWS dataset files.

    This function performs the following steps:
    1. Extract `.csv` files to `.txt` files.
    2. Clean the data by removing unnecessary characters and labels.
    3. Delete the original `.csv` files after extraction.

    :return: None
    """
    logging.basicConfig(level=logging.INFO)
    for filename in os.listdir(data_dir + f"/{DATASET_NAME}"):
        if filename.endswith(".csv"):
            full_path = os.path.join(data_dir, DATASET_NAME, filename)

            # Read the .csv file
            with open(full_path, 'r') as f_in:
                data = f_in.read()

            # Clean the data by removing unnecessary labels and characters
            cleaned_data = re.sub(r'"\d+",', '', data)
            cleaned_data = cleaned_data.replace('"', '').replace('\\', '').strip()

            # Save the cleaned data to a .txt file
            cleaned_file_path = full_path.replace(".csv", ".txt")
            with open(cleaned_file_path, 'w') as f_out:
                f_out.write(cleaned_data)

            # Remove the original .csv file
            os.remove(full_path)

    logging.info("Extraction and cleaning complete.")
