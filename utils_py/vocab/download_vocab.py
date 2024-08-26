import sentencepiece as spm
import os


def download_vocab(data_dir):
    """
    Download and process the vocabulary from the specified datasets.

    Args:
        data_dir (str): The directory to store the vocabulary file.

    Returns:
        str: Path to the created vocabulary file.
    """
    vocab_file_path = os.path.join(data_dir, "vocab_20000_words.txt")

    print("Creating and populating the vocab file with datasets like AG_NEWS and WikiText.")
    with open(vocab_file_path, "w") as f:
        for file_name in ["train", "valid", "test"]:
            with open(f"data/wikitext/{file_name}.txt", "r") as wikitext_file:
                f.write(wikitext_file.read().lower())
        for file_name in ["train", "test"]:
            with open(f"data/ag_news/{file_name}.txt", "r") as ag_news_file:
                f.write(ag_news_file.read().lower())

    return vocab_file_path


def train_vocab(vocab_file_path):
    """
    Train the SentencePiece model using the specified vocabulary file.

    Args:
        vocab_file_path (str): Path to the vocabulary file.

    Returns:
        None
    """
    print("Starting SentencePiece vocabulary training...")
    command_string = (
        f"--input={vocab_file_path} "
        f"--model_prefix=spm "
        f"--vocab_size=20000 "
        f"--model_type=word "
        f"--character_coverage=0.9995"
    )
    spm.SentencePieceTrainer.Train(command_string)
    print("SentencePiece vocabulary training completed.")
