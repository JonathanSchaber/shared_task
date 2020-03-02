import argparse
import json
import csv
import torch
from torch import nn
from bigram_based_models import load_id_to_label, load_id_to_repr, load_data, write_to_file, parse_cmd_args


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--path_train", type=str, help="Path to train data.")
    parser.add_argument("-d", "--path_dev", type=str, help="Path to dev data.")
    parser.add_argument("-o", "--path_out", type=str, help="Path to output file containing predictions.")
    parser.add_argument("-c", "--path_config", type=str, help="Path to hyperparamter/config file (json).")
    return parser.parse_args()


def load_config(path):
    """Load configuration from json.

    Args:
        path: str
    """
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def load_batches(csv_reader, batch_size, granularity):
    """
    Lazy loading of batches. Returns

    Args:
        path_train_data: str
        batch_size: int
    """
    i = 0
    if granularity == "binary":
        index = 0 + 3
    elif granularity == "ternary":
        index = 1 + 3
    elif granularity == "finegrained":
        index = 2 + 3
    else:
        raise Exception('ERROR: Granularity level not known.')
    batch = []

    while i < batch_size:
        try:
            row = next(csv_reader)
            batch.append((row[1], row[index]))
        except StopIteration:
            break

    return batch if batch else None


def train_model(config):
    batch_size = config['batch_size']
    granularity = config['granularity']
    path_train = config['path_train']
    num_epochs = config['num_epochs']

    for epoch in range(1, num_epochs + 1):
        train_reader = csv.reader(open(path_train, 'r', encoding='utf8'))
        for batch_num, batch in enumerate(load_batches(csv_reader=train_reader, batch_size=batch_size, granularity=granularity)):





class Model(torch.nn):

    def __init__(self, char_to_row, embedding_dim):

        self.embedding = nn.Embedding(len(char_to_row), embedding_dim=embedding_dim)
        self.char_lang_model = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)


def main():
    args = parse_cmd_args()
    config = load_config(args.path_config)
    batch_iterator = yield_batches(args.path_train, config['batchsize'])
    trained_model = train_model(config, batch_iterator, criterion)