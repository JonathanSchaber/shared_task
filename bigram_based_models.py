import csv
import numpy as np
import argparse
from sklearn.svm import SVC

"""
Train 'classical' models on bigram based representations.
"""


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--path_train", type=str, help="Path to train data.")
    parser.add_argument("-d", "--path_dev", type=str, help="Path to dev data.")
    parser.add_argument("-o", "--path_out", type=str, help="Path to output file containing predictions.")
    return parser.parse_args()


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
            repr = np.array([int(float(i)) for i in columns[4:]], dtype=int)
            X_list.append(repr)
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
        path_train: str
        path_dev: str
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


def write_to_file(predictions, y_dev, granularity, path_out):
    """Write predictions to file.

    Args:
        predictions: numpy-array with labels (0 or 1)
    """
    text_ids = []
    with open('data/main/dev_main.csv', 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            text_ids.append(row[0])
    with open(path_out.rstrip('.csv') + '_' + granularity + '.csv', 'w', encoding='utf8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["text_id", "label_pred", "label_true", "granularity_index"])
        for text_id, label_pred, label_true in zip(text_ids, predictions, y_dev):
            csv_writer.writerow([text_id, label_pred, label_true, granularity])


def main():
    print('Parse cmd args...')
    args = parse_cmd_args()
    for granularity in ["binary", "ternary", "finegrained"]:
        print('Load training and test data...')
        X_train, y_train, X_dev, y_dev = load_data(args.path_train, args.path_dev, granularity)
        print('Train svm...')
        svm = train_svm(args, X_train, y_train)
        print('Predict on dev-set...')
        predictions = svm_predict(svm, X_dev)
        print('Write to output file...')
        write_to_file(predictions, y_dev, granularity, args.path_out)


if __name__ == '__main__':
    main()
