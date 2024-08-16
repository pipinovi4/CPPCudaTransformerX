from datasets import load_dataset

DATASET_NAME = "wikitext"


def download_wikitext():
    """
    Load the WikiText-2 dataset.

    Returns:
    dataset (datasets.Dataset): The WikiText-2 dataset.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    return dataset
