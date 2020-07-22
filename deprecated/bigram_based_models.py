import os
import re
import csv
import numpy as np
import argparse
from sklearn.svm import SVC
from utils import get_timestamp


"""
Train 'classical' models on bigram based representations.
"""


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--path_train", type=str, help="Path to train data.")
    parser.add_argument("-d", "--path_dev", type=str, help="Path to dev data.")
    parser.add_argument("-o", "--path_out_dir", type=str, help="Path to output directory.")
    return parser.parse_args()


def get_xperc(args):
    """Get the number of examples per class.

    Args:
        args: argparse-argument object
    """
    result = re.search(r'(\d+)\.csv$', args.path_train)
    if result:
        return int(result.group(1))
    else:
        msg = 'Error: Number of examples per class could not be read from train_path. train_path: {}'
        raise Exception(msg.format(args.path_train))


def load_dataset(path, granularity):
    """Load mapping of text-ids to bigram representations.

    Args:
        path: str
        granularity: str
    Return: Tuple containing
        X: 2D-Array
        y: 1D-Array
    """
    X_list = []
    y_list = []
    gran_to_idx = {
        'binary': 1,
        'ternary': 2,
        'finegrained': 3
    }
    print('Loading data from file...')
    with open(path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            columns = line.strip('\n').split(', ')
            # text_id, label_binary, label_ternary, label_finegrained = columns[:4]
            text_repr = np.array([int(float(i)) for i in columns[4:]], dtype=int)
            X_list.append(text_repr)
            y_list.append(columns[gran_to_idx[granularity]])
            if i % 10000 == 0 and i != 0:
                print('Loaded 10000 rows from file...')
    print('Converting lists to arrays...')
    X_train = np.array(X_list, dtype=int)
    y_train = np.array(y_list, dtype=int)
    return X_train, y_train


def load_data(path_trainset, path_devset, granularity):
    """Load training data into numpy arrays.

    Args:
        path_trainset: str
        path_devset: str
        granularity: str
    Return: Tuple containing:
        X_train: 2D-Array
        y_train: 1D-Array
        X_dev: 2D-Array
        y_dev: 1D-Array
    """
    print('Loading trainset...')
    X_train, y_train = load_dataset(path_trainset, granularity)
    print('Loading devset...')
    X_dev, y_dev = load_dataset(path_devset, granularity)
    return X_train, y_train, X_dev, y_dev


def train_svm(args, X, y):
    """Train support vector machine.

    Args:
        args: argparse-parser-object
        X: 2D-Array, training-data
        y: 1D-Array, test-data
    Return:
        trained sklearn-svm object
    """
    clf = SVC(gamma='auto')
    clf.fit(X, y)
    return clf


def svm_predict(svm, X_dev):
    """Predict with trained svm on the dev-set for evaluation.

    Args:
        svm: sklearn-svm-object
        X_dev: 2D-numpy array
    Return:
        1D-numpy array
    """
    return svm.predict(X_dev)


def write_to_file(predictions, y_dev, granularity, path_dev, path_out_dir, xperc):
    """Write predictions to file.

    Args:
        predictions: numpy-array with labels
        y_dev: numpy-array with true labels
        granularity: str
        path_dev: str
        path_out_dir: str
        xperc: int, number of examples per class
    """
    # Load text-ids
    text_ids = []
    with open(path_dev, 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            text_ids.append(row[0])
    # write results to file
    fname_out = 'results_{granularity}_{xperc}_{timestamp}.csv'
    fname_out = fname_out.format(granularity=granularity, xperc=xperc, timestamp=get_timestamp())
    with open(os.path.join(path_out_dir, fname_out), 'w', encoding='utf8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["text_id", "label_pred", "label_true", "granularity"])
        for text_id, label_pred, label_true in zip(text_ids, predictions, y_dev):
            csv_writer.writerow([text_id, label_pred, label_true, granularity])


def main():
    print('Parse cmd args...')
    args = parse_cmd_args()
    xperc = get_xperc(args)
    for granularity in ["binary", "ternary", "finegrained"]:
        print('Load training and test data...')
        X_train, y_train, X_dev, y_dev = load_data(args.path_train, args.path_dev, granularity)
        print('Train svm...')
        svm = train_svm(args, X_train, y_train)
        print('Predict on dev-set...')
        predictions = svm_predict(svm, X_dev)
        print('Write to output file...')
        write_to_file(predictions, y_dev, granularity, args.path_dev, args.path_out_dir, xperc)
        print('Done.')


if __name__ == '__main__':
    main()
