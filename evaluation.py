import argparse
import csv


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gold_path', type=str, default='data/main/dev_main.csv', help='Path to gold labels (dev_main.csv).')
    parser.add_argument('-o', '--path_out', type=str, default='data/main/results.csv',
                        help='Path to output file to write results.')
    parser.add_argument('-p', '--predicted_path', type=str, default='data/main/predicted.csv',
                        help='Path to predicted lables (predicted.csv).')
    return parser.parse_args()


def process_predictions_file(pred_file):
    """Processes pred_file with the following structure:
    text_id, label_binary, label_ternary, label_finegrained, pred_binary, pred_ternary, pred_finegrained, text, masked, source

    Args:
        pred_file: str
    Return:
        None
    """
    bin_true_pos = 0
    bin_false_pos = 0
    bin_true_neg = 0
    bin_false_neg = 0

    ter_true = 0
    ter_false = 0
    fine_true = 0
    false_fine = 0

    TER = False
    FINE = False

    with open(pred_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for i, line in enumerate(csv_reader):
            text_id, label_binary, label_ternary, label_finegrained, pred_binary, pred_ternary, pred_finegrained, text, masked, source = line
            if i == 0:
                if not pred_ternary == "NULL": TER = True 
                if not pred_finegrained == "NULL": FINE = True 

            if label_binary == "1" and pred_binary == "1": bin_true_pos += 1
            elif label_binary == "1" and pred_binary == "0": bin_false_neg += 1
            elif label_binary == "0" and pred_binary == "1": bin_false_pos += 1
            elif label_binary == "0" and pred_binary == "0": bin_true_neg += 1

            if TER: 
                if label_ternary == pred_ternary: ter_true += 1 
                else: ter_false += 1
            if FINE:
                if label_finegrained == pred_finegrained: fine_true += 1 
                else: false_fine += 1
    
    bin_prec = bin_true_pos / (bin_true_pos + bin_false_pos)
    bin_rec = bin_true_pos / (bin_true_pos + bin_false_neg)
    bin_acc = (bin_true_pos + bin_true_neg) / (bin_true_neg + bin_false_pos + bin_true_neg + bin_false_neg)
    bin_f1 = 2 * (bin_prec * bin_rec) / (bin_prec + bin_rec)
    
    print("=============== RESULTS BINARY ===============")
    print("Precision: " + str(100 * bin_prec) + "%")
    print("Recall: " + str(100 * bin_rec) + "%")
    print("Accuracy: " + str(100 * bin_acc) + "%")
    print("F1: " + str(100 * bin_f1))
    print("==============================================\n")

    if TER:
        if ter_true != 0:
            print('Accuracy ternary: {}'.format((ter_true/(ter_true + ter_false)*100)))
        else:
            print('No correct ternary predictions...')
    if FINE:
        if fine_true != 0:
            print('Accuracy finegrained: {}'.format((fine_true/(fine_true + false_fine)*100)))
        else:
            print('No correct finegrained predictions...')
    

def compare_line_by_line(gold_path, predicted_path):
    """Compares the gold lables to the predicted ones.
 
    Args:
        gold_path: str, path to gold csv file
        predicted_path: str, path to predicted csv file
    Return:
        true_pos, false_pos, true_neg, false_neg: int
    """
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
 
    gold_labels = csv.reader(open(gold_path, 'r', encoding='utf-8'))
    predicted_labels = csv.reader(open(predicted_path, 'r', encoding='utf-8'))
    for line_gold, line_pred in zip(gold_labels, predicted_labels):
        if line_gold[0] != line_pred[0]:
            raise Error
        gold = int(float(line_gold[3]))
        pred = int(float(line_pred[1]))
        if gold == 1 and pred == 1:
            true_pos += 1
        elif gold == 1 and pred == 0:
            false_neg += 1
        elif gold == 0 and pred == 1:
            false_pos += 1
        elif gold == 0 and pred == 0:
            true_neg += 1
        else:
            print('Error unallowed values for gold: ' + str(gold) + ' or pred: ' + str(pred))
    return true_pos, false_pos, true_neg, false_neg
     

def write_measures_to_file(true_pos, false_pos, true_neg, false_neg, out_file):
    """Writes the counts from compare_line_by_line() to the outfile
        Returns none.
 
    Args:
        true_pos, false_pos, true_neg, false_neg: int
    """

    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write('True Positives: ' + str(true_pos) + '\n'
                      'False Positives: ' + str(false_pos) + '\n'
                      'True Negatives: ' + str(true_neg) + '\n'
                      'False Negatives: ' + str(false_neg) + '\n\n'
                      'Total: ' + str(true_pos + false_pos + true_neg + false_neg))


def main():
    args = parse_cmd_args()
    #true_pos, false_pos, true_neg, false_neg = compare_line_by_line(args.gold_path, args.predicted_path)
    #write_measures_to_file(true_pos, false_pos, true_neg, false_neg, args.path_out)
    process_predictions_file(args.predicted_path)


if __name__ == '__main__':
    main()
