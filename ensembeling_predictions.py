import argparse
import csv
import numpy as np

from collections import defaultdict
from statistics import mode


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', type=str, action='append', nargs='+', help='path to prediction files')
    parser.add_argument('-o', '--output_file', type=str, help='path to prediction files')
    parser.add_argument('-m', '--modus', type=str, help='modus: "dev" or "test"')
    return parser.parse_args()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def majority_decision_dev(list_files):
    """Read in the different predictions and compute definitive output.

    Args:
        list_files: list
        outfile: str
    Return:
        definite_preds: list
    """
    num_files = len(list_files)
    id_preds = defaultdict(lambda: [])
    definite_preds = []

    for file in list_files:
        with open(file, "r", encoding="utf8") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                text_id, label_binary, label_ternary, label_finegrained, pred_binary, pred_ternary, pred_finegrained, text, masked, source = row
                id_preds[text_id].append(pred_binary)

    assert all(len(labels_pred) == num_files for labels_pred in id_preds.values()), "ATTENTION: Something wrong with prediction files!"

    with open(list_files[0], "r", encoding="utf8") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            text_id, label_binary, label_ternary, label_finegrained, pred_binary, pred_ternary, pred_finegrained, text, masked, source = row
            definite_preds.append([text_id, label_binary, label_ternary, label_finegrained, mode(id_preds[text_id]), pred_ternary, pred_finegrained, text, masked, source])

    return definite_preds
    

def majority_decision_test(list_files):
    """Read in the different predictions and compute definitive output.

    Args:
        list_files: list
        outfile: str
    Return:
        definite_preds: list
    """
    num_files = len(list_files)
    id_preds = defaultdict(lambda: [])
    definite_preds = []

    for file in list_files:
        with open(file, "r", encoding="utf8") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                tweet_id, label_pred, confidence = row
                id_preds[tweet_id].append((label_pred, confidence))

    assert all(len(labels_pred) == num_files for labels_pred in id_preds.values()), "ATTENTION: Something wrong with prediction files!"
                
    for tweet_id in id_preds:
        maj_pred = mode([pred[0] for pred in id_preds[tweet_id]])
        maj_conf = softmax([sum([pred[1] for pred in id_preds.values() if pred[0] == maj_pred]), sum([pred[1] for pred in id_preds.values() if pred[1] != maj_pred])])[0]
        definite_preds.append([tweet_id, maj_pred, maj_conf])

    return definite_preds


def write_to_file(preds, outfile):
    """Write definite predictions to file.
    
    Args:
        preds: list
        outfile: str
    Return:
        None
    """
    with open(outfile, "w", encoding="utf8") as f:
        csv_writer = csv.writer(f)
        for row in preds:
            csv_writer.writerow(row)


def main():
    print('Parse cmd line args...')
    args = parse_cmd_args()
    list_files = args.input_files[0]
    output_file = args.output_file
    modus = args.modus
    if len(list_files) % 2 == 0:
        print("Warning: Even number of files provided!")
    if modus == "dev":
        print("Calculating majority decisions...")
        maj_preds = majority_decision_dev(list_files)
    elif modus == "test":
        print("Calculating majority decisions...")
        maj_preds = majority_decision_test(list_files)
    else:
        print("Warning: Unknown Modus! Aborting!")
        return False
    print("Writing to file...")
    write_to_file(maj_preds, output_file)


if __name__ == '__main__':
    main()
