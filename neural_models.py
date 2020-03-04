import os
import argparse
import json
import csv
import numpy as np
import torch
from torch import nn
from utils import get_timestamp


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
        csv_reader: csv-reader object
        batch_size: int
        granularity: str
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

    while True:
        batch = []
        while i < batch_size:
            try:
                row = next(csv_reader)
                batch.append((row[1], row[index]))
            except StopIteration:
                return
        yield batch if batch else None


def get_num_batches(path_train, batch_size):
    """Get the number of batches for one epoch.

    Args:
        path_train: str
        batch_size: int
    """
    with open(path_train, 'r', encoding='utf8') as f:
        num_examples = 0
        for _ in f:
            num_examples += 1
        num_batches = num_examples / batch_size
        if num_examples % batch_size != 0:
            num_batches += 1
        return num_batches


def load_char_to_idx():
    with open('char_to_idx.json', 'r', encoding='utf8') as f:
        return json.load(f)


def create_char_to_idx(path_train):
    """Create a json file mapping characters to idxs.

    Args:
        path_train: str
    """
    i = 0
    char_to_idx = {}
    reader = csv.reader(open(path_train, 'r', encoding='utf8'))
    for row in reader:
        text = row[1]
        for char in text:
            if char in char_to_idx:
                continue
            else:
                char_to_idx[char] = i
                i += 1
    with open('char_to_idx.json', 'w', encoding='utf8') as f:
        json.dump(char_to_idx, f)
    idx_to_char = {val: key for key, val in char_to_idx.items()}
    with open('idx_to_char.json', 'w', encoding='utf8') as f:
        json.dump(idx_to_char, f)


def train_model(config):
    batch_size = config['batch_size']
    granularity = config['granularity']
    path_train = config['path_train']
    num_epochs = config['num_epochs']
    lr = config['learning_rate']
    num_classes = config['num_classes']
    dropout = config['dropout']
    hidden_gru_size = config['hidden_gru_size']
    num_gru_layers = config['num_gru_layers']
    if not os.path.exists('char_to_idx.json'):
        create_char_to_idx(path_train)
    char_to_idx = load_char_to_idx()
    model = SeqToLabelModel(char_to_idx, embedding_dim=config['embedding_dim'], hidden_gru_size=hidden_gru_size,
                            num_gru_layers=num_gru_layers, num_classes=num_classes, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_batches = get_num_batches(path_train, batch_size)

    for epoch in range(1, num_epochs + 1):
        train_reader = csv.reader(open(path_train, 'r', encoding='utf8'))
        losses = []
        for batch_num, batch in enumerate(load_batches(csv_reader=train_reader, batch_size=batch_size,
                                                       granularity=granularity)):
            if not batch:
                print('WARNING: Empty batch.')
                continue
            x_train = torch.Tensor(batch[0], dtype=torch.LongTensor).to(device)
            y_train = torch.Tensor(batch[1], dtype=torch.LongTensor).to(device)

            # zero the gradients
            optimizer.zero_grad()

            # propagate forward
            output = model(x_train)

            # compute loss
            loss = criterion(output, y_train)
            losses.append(loss)

            # backward propagation
            loss.backward()

            # optimize
            optimizer.step()

            if batch_num % 100 == 0:
                avg_loss = np.mean(losses)
                msg = 'Epoch [{}/{}], batch [{}/{}], avg. loss: {}'
                print(msg.format(epoch, num_epochs, batch_num, num_batches, avg_loss))
                losses = []

    return model


class SeqToLabelModel(torch.nn.Module):

    def __init__(self, char_to_idx, embedding_dim, hidden_gru_size, num_gru_layers, num_classes, dropout):
        super(SeqToLabelModel, self).__init__()
        self.embedding = nn.Embedding(len(char_to_idx), embedding_dim=embedding_dim)
        self.char_lang_model = nn.GRU(input_size=embedding_dim, hidden_size=hidden_gru_size,
                                      num_layers=num_gru_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_gru_size, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        seq_output = self.char_lang_model(embeds)
        output = self.linear(seq_output)
        return output


def save_model(trained_model, config, use_server_paths):
    if use_server_paths:
        path_out = '/home/user/jgoldz/storage/shared_task/models'
    else:
        path_out = 'models'
    fname = '{model_name}_{config_id}_{timestamp}.model'.format(model_name=config['model_name'],
                                                                config_id=config['config_id'],
                                                                timestamp=get_timestamp())
    fpath = os.path.join(path_out, fname)
    torch.save(trained_model, fpath)
    print('Model saved to {}'.format(fpath))


def main():
    print('Parse cmd line args...')
    args = parse_cmd_args()
    print('Loading config from {}...'.format(args.path_config))
    config = load_config(args.path_config)
    print('Start model training')
    trained_model = train_model(config)
    print('Saving trained model...')
    save_model(trained_model, config, args.server)


if __name__ == '__main__':
    main()