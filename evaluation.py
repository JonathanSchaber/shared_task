import argparse
import csv


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gold_path', type=str, default='data/main/dev_main.csv', help='Path to gold labels (dev_main.csv).')
    parser.add_argument('-o', '--path_out', type=str, default='data/main/results.csv',
                        help='Path to output file to write results.')
    parser.add_argument('-p', '--predicted_path', type=str default='data/main/predicted.csv',
                        help='Path to predicted lables (predicted.csv).')
    return parser.parse_args()


def compare_line_by_line(gold_path, predicted_path):
    """Compares the gold lables to the predicted ones.
        Returns the counts.
 
    Args:
        gold_path: str, path to gold csv file
        predicted_path: str, path to predicted csv file
    Return:
        TODO
    """
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
 
    gold_labels = csv.reader(open(gold_path, 'r', encoding='utf8') )
    predicted_labels = csv.reader(open(predicted_path, 'r', encoding='utf8'))
    for line_gold, line_pred in zip(gold_labels, predicted_labels):
        if line_gold[0] != line_pred[0]:
            raise Error
        gold = line_gold[3]
        pred = line_pred[1]
        if gold == 1 and pred == 1:
            true_pos += 1
        elif gold == 1 and pred == 0:
            false_ned += 1
        elif gold == 0 and pred == 1:
            false_pos += 1
        elif gold == 0 and pred == 0:
            true_neg += 1
    return true_pos, false_pos, true_neg, false_neg
     

def write_measures_to_file(true_pos, false_pos, true_neg, false_neg, out_file):
    outfile = 


def main():
    args = parse_cmd_args()
    true_pos, false_pos, true_neg, false_neg = compare_line_by_line(args.gold_path, args.predicted_path)
    write_measures_to_file(true_pos, false_pos, true_neg, false_neg, arg.path_out)


if __name__ == '__name__':
    main()
