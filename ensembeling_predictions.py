import argparse
import csv

from collections import defaultdict
from statistics import mode

def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', type=str, action='append', nargs='+', help='path to prediction files')
    parser.add_argument('-o', '--output_file', type=str, help='path to prediction files')
    return parser.parse_args()


def majority_decision(list_files):
    """Read in the different predictions and compute definitive output.

    Args:
        list_files: list
        outfile: str
    Return:
        definite_preds: list
    """
    num_files = len(list_files)
    id_text = {}
    id_preds = defaultdict(lambda: [])
    definite_preds = []

    for file in list_files:
        with open(file, "r", encoding="utf8") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                tweet_id, text, label_pred = row
                id_text[tweet_id] = text
                id_preds[tweet_id].append(label_pred)

    assert all(len(labels_pred) == num_files for labels_pred in id_preds.values())
                
    for tweet_id in id_preds:
        definite_preds.append([tweet_id, mode(id_preds[tweet_id]), id_text[tweet_id]])

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
    if len(list_files) % 2 != 0:
        print("Warning: Even number of files provided!")
    print("Calculating majority decisions...")
    maj_preds = majority_decision(list_files)
    print("Writing to file...")
    write_to_file(preds, output_file)


if __name__ == '__main__':
    main()
