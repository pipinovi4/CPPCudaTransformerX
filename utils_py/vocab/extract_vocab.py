import os


def is_not_numeric(sentence):
    """
    Check if the sentence is not composed only of numeric characters.

    Args:
        sentence (str): The sentence to check.

    Returns:
        bool: True if the sentence is not numeric, False otherwise.
    """
    return not sentence.strip().isdigit()


def does_not_start_with_special_char(word, special_chars):
    """
    Check if the word does not start with any of the specified special characters or a number.

    Args:
        word (str): The word to check.
        special_chars (list): A list of special characters to check for.

    Returns:
        bool: True if the word does not start with any special character or number, False otherwise.
    """
    return bool(word) and not (word[0].isdigit() or word.startswith(tuple(special_chars)))


def split_and_filter_word(word, special_chars):
    """
    Split words by commas and filter out unwanted tokens.

    Args:
        word (str): The word to split and filter.
        special_chars (list): A list of special characters to check for.

    Returns:
        list: A list of filtered words.
    """
    return [
        part for part in word.split(',')
        if does_not_start_with_special_char(part, special_chars) and is_not_numeric(part)
    ]


def extract_vocab(data_dir):
    """
    Extract and filter the vocabulary from the SentencePiece model output.

    Args:
        data_dir (str): The directory where the vocabulary file is located.

    Returns:
        None
    """
    vocab_file_path = os.path.join(data_dir, "vocab_20000_words.txt")

    with open("spm.vocab", "r") as spm_vocab_file:
        lines = spm_vocab_file.readlines()

    special_chars = ['-', '#', '&', '@', '$', '%', '^', '*', '+', '=', '<', '>', '?', '!', '~', '`', '|', '\\', '/', '(', ')', '[', ']', '{', '}', ':', ';', '"', "'", '.']

    words = []
    for line in lines:
        word = line.split('\t')[0].replace('▁', '')
        if not line.startswith('<'):
            words.extend(split_and_filter_word(word, special_chars))

    with open(vocab_file_path, "w") as output_file:
        output_file.write('\n'.join(words))

    os.remove("spm.vocab")
    os.remove("spm.model")

    print("Filtered vocabulary saved and unnecessary files deleted.")
