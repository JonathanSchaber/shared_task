import csv
import argparse
import os
from random import shuffle


"""
To execute the script use: 
python3 split_corpus.py -r 0.8
to get a train test split of 0.8 using the default paths.
"""


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--path_in", type=str, default='data/main/main.csv', help="Path to main.csv")
    parser.add_argument("-o", "--path_out", type=str, default='data/main/', help="Path to output directory.")
    parser.add_argument('-r', '--ratio', type=float, help='Train-dev ratio.')
    return parser.parse_args()


def load_corpus(path_in):
    """Load main corpus from <path_in>
    
    Args:
        path_in: str
    """
    with open(path_in, 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f)
        return [row for row in csv_reader]


def get_split_index(ratio, num_examples):
    """Compute index at which corpus is split.

    Args:
        ratio: float, for example 0.7 -> 70% training ste, 30% dev set
        num_examples: int
    """
    return int(ratio * num_examples)


def write_csv_to_file(rows, path):
    """Write rows to csv-file at <path>

    Args:
        rows: list of list of any
        path: str
    """
    with open(path, 'w', encoding='utf8') as f:
        csv_writer = csv.writer(f)
        for row in rows:
            csv_writer.writerow(row)


def create_train_dev_files(corpus, split_index, path_out):
    """Write examples to train and dev files.

    Args:
        corpus: list of rows
        split_index: int
        path_out: str
    """
    train_corpus = corpus[:split_index]
    dev_corpus = corpus[split_index:]
    write_csv_to_file(train_corpus, path=os.path.join(path_out, 'train_main.csv'))
    write_csv_to_file(dev_corpus, path=os.path.join(path_out, 'dev_main.csv'))


def main():
    print('Parse command line arguments...')
    args = parse_cmd_args()
    print('Load corpus...')
    corpus = load_corpus(args.path_in)
    print('Shuffle corpus...')
    shuffle(corpus)
    print('Get split index...')
    split_index = get_split_index(args.ratio, len(corpus))
    print('Write train-dev corpora to file...')
    create_train_dev_files(corpus, split_index, args.path_out)


if __name__ == '__main__':
    main()
