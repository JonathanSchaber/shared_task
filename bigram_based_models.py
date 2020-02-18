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


def load_data(path_train, path_dev):
    """Load training data into numpy arrays.

    Args:
        path_train: str
        path_dev: str
    Return: Tuple containing:
        X_train: 2D-Array
        y_train: 1D-Array
        X_dev: 2D-Array
        y_dev: 1D-Array
    """
    # Loading of train set
    # load id-to-label mappings
    id_to_label = {}
    with open('data/main/train_main.csv', 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f)
        for text_id, text, masked, label, corpus in csv_reader:
            id_to_label[text_id] = label
    # load id to repr mappings
    id_to_repr = {}
    with open('data/main/train_main_bigr_repr.csv', 'r', encoding='utf8') as f:
        line_counter = 0
        max_line = 2000
        for line in f:
            if line_counter > max_line:
                break
            columns = line.strip('\n').split(', ')
            text_id, repr = columns[0], np.array([int(i) for i in columns[1:]])
            id_to_repr[text_id] = repr
            line_counter += 1

    num_feats = len(id_to_repr[text_id])
    num_examples = len(id_to_repr)

    # convert to correct matrix/column vector
    X_train = np.zeros((num_examples, num_feats))
    y_train = np.zeros(num_examples)

    for i, text_id in enumerate(id_to_repr):
        label = id_to_label[text_id]
        repr = id_to_repr[text_id]
        X_train[i] = repr
        y_train[i] = label

    # Loading of testset


    return X_train, y_train, X_dev, y_dev



def train_svm(X, y):
    """Train support vector machine.

    Args:
        X: 2D-Array, training-data
        y: 1D-Array, test-data
    Return:
        trained sklearn-svm object
    """
    clf = SVC(gamma='auto')
    clf.fit(X, y)
    print(clf.predict([[-0.8, -1]]))



def svm_predict_svm(svm, X_dev, y_dev):
    """Predict with trained svm on the dev-set for evaluation."""
    pass


def write_to_file(predictions, path_out):
    """Write predictions to file.

    Args:
        predictions: numpy-array with labels (0 or 1)
    """


def main():
    print('Parse cmd args...')
    args = parse_cmd_args()
    print('Load training and test data...')
    X_train, y_train, X_dev, y_dev = load_data(args.path_train, args.path_dev)
    print('Train svm...')
    svm = train_svm(args, X_train, y_train)
    print('Predict on dev-set...')
    predictions = svm_predict(svm, X_dev, y_dev)
    print('Write to output file...')
    write_to_file(predictions, args.path_out)


if __name__ == '__main__':
    main()