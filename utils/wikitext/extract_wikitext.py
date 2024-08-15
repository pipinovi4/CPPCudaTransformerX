import os


# Extracts the wikitext from the wikitext-103 dataset and saves it to a file
def extract_wikitext(dataset, data_dir):
    """
    Extract the WikiText dataset files and save.
    """

    def save_split_to_file(split, filename):
        """
        Save the split to a file.

        :param split:
        :param filename:
        :return:
        """
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in split:
                f.write(entry['text'] + '\n')

    os.makedirs(data_dir, exist_ok=True)
    save_split_to_file(dataset['train'], os.path.join(data_dir, "train.txt"))
    save_split_to_file(dataset['validation'], os.path.join(data_dir, "valid.txt"))
    save_split_to_file(dataset['test'], os.path.join(data_dir, "test.txt"))
