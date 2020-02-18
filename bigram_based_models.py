import csv
import numpy as np
import argparse
from sklearn.svm import SVC


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--path_train", type=str, default='data/main/main.csv', help="Path to train data.")
    parser.add_argument("-d", "--path_dev", type=str, default='data/main/main.csv', help="Path to dev data.")
    parser.add_argument("-o", "--path_out", type=str, default='data/main/predicted.csv',
                        help="Path to output file containing predictions.")
    return parser.parse_args()


def load_id_to_label(path):
    """Load mapping of text-ids to labels.

    Args:
        path: str
    """
    id_to_label = {}
    with open(path, 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f)
        for text_id, text, masked, label, corpus in csv_reader:
            id_to_label[text_id] = label
    return id_to_label


def load_id_to_repr(path, limit=None):
    """Load mapping of text-ids to bigram representations.

    Args:
        path: str
        limit: int
    Return: Tuple containing
        id_to_repr: {text-id<str>: multi-hot-representation<ndarray>}
        id_list_ordered: list of str
    """
    id_to_repr = {}
    id_list_ordered = []
    line_counter = 0
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            if limit is not None:
                if line_counter > limit:
                    break
            columns = line.strip('\n').split(', ')
            text_id, repr = columns[0], np.array([int(i) for i in columns[1:]])
            id_to_repr[text_id] = repr
            id_list_ordered.append(text_id)
            line_counter += 1
    return id_to_repr, id_list_ordered


def load_data(path_train, path_dev):
    """Load training data into numpy arrays.

    Args:
        path_train: str
        path_dev: str
    Return: Tuple containing:
        X_train: 2D-Array
        y_train: 1D-Array
        X_dev: 2D-Array
    """
    # Loading of train set
    id_to_label_train = load_id_to_label('data/main/train_main.csv')
    id_to_repr_train, id_list_ordered_train = load_id_to_repr('data/main/train_main_bigr_repr.csv')

    text_id = list(id_to_repr_train.keys())[0]
    num_feats = len(id_to_repr_train[text_id])
    num_examples_train = len(id_to_repr_train)

    # convert to correct matrix/column vector
    X_train = np.zeros((num_examples_train, num_feats))
    y_train = np.zeros(num_examples_train)

    for i, text_id in enumerate(id_list_ordered_train):
        label = id_to_label_train[text_id]
        repr = id_to_repr_train[text_id]
        X_train[i] = repr
        y_train[i] = label

    # Loading of dev set
    id_to_repr_dev, id_list_ordered_dev = load_id_to_repr('data/main/dev_main_bigr_repr.csv', limit=1000)
    num_examples_dev = len(id_to_repr_dev)

    X_dev = np.zeros((num_examples_dev, num_feats))
    for i, text_id in enumerate(id_list_ordered_dev):
        repr = id_to_repr_dev[text_id]
        X_dev[i] = repr

    return X_train, y_train, X_dev



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


def write_to_file(predictions, path_out):
    """Write predictions to file.

    Args:
        predictions: numpy-array with labels (0 or 1)
    """
    text_ids = []
    with open('data/main/dev_main.csv', 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            text_ids.append(row[0])
    with open(path_out, 'w', encoding='utf8') as f:
        csv_writer = csv.writer(f)
        for text_id, label in zip(text_ids, predictions):
            csv_writer.writerow([text_id, label])


def main():
    print('Parse cmd args...')
    args = parse_cmd_args()
    print('Load training and test data...')
    X_train, y_train, X_dev = load_data(args.path_train, args.path_dev)
    print('Train svm...')
    svm = train_svm(args, X_train, y_train)
    print('Predict on dev-set...')
    predictions = svm_predict(svm, X_dev)
    print('Write to output file...')
    write_to_file(predictions, args.path_out)


if __name__ == '__main__':
    main()