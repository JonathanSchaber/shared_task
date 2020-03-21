import os

from predict import load_model, get_num_examples, predict_on_input
from corpus_parser import Parser
from swiss_char_checker import check_sentences

# General Pipeline: specify model -> specify test set -> specify if with char-checker -> specify if written to file

def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--path_model", type=str, help="Path to model.")
    parser.add_argument("-t", "--path_test", type=str, help="Path to test data.")
    parser.add_argument("-c", "--path_config", type=str, help="Path to hyperparamter/config file (json).")
    parser.add_argument("-cc", "--charchecker", action="store_true", default=True, help="Do pre-elimination with char-checker.")
    parser.add_argument("-w", "--write", action="store_true", default=False, help="Write to file (name automatically generated.")
    return parser.parse_args()


def process_predictions_file(pred_file):
    """Processes pred_file with the following structure:
    text_id, label_binary, label_ternary, label_finegrained, pred_binary, pred_ternary, pred_finegrained, text, masked, source

    Args:
        pred_file: str
    Return:
        None
    """
    false_preds = []

    bin_true_pos = 0
    bin_false_pos = 0
    bin_true_neg = 0
    bin_false_neg = 0

    ter_true = 0
    ter_false = 0
    fine_true = 0
    fine_false = 0

    TER = False
    FINE = False

    with open(pred_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for i, line in enumerate(csv_reader):
            text_id, label_binary, label_ternary, label_finegrained, pred_binary, pred_ternary, pred_finegrained, text, masked, source = line
            if i == 0:
                if not pred_ternary == "NULL": TER = True 
                if not pred_finegrained == "NULL": FINE = True 

            if label_binary == "0" and pred_binary == "0":
                bin_true_pos += 1
            elif label_binary == "0" and pred_binary == "1":
                bin_false_neg += 1
                false_preds.append(line)
            elif label_binary == "1" and pred_binary == "0":
                bin_false_pos += 1
                false_preds.append(line)
            elif label_binary == "1" and pred_binary == "1":
                bin_true_neg += 1

            if TER: 
                if label_ternary == pred_ternary: ter_true += 1 
                else: ter_false += 1
            if FINE:
                if label_finegrained == pred_finegrained: fine_true += 1 
                else: fine_false += 1
    
    bin_prec = bin_true_pos / (bin_true_pos + bin_false_pos)
    bin_rec = bin_true_pos / (bin_true_pos + bin_false_neg)
    bin_acc = (bin_true_pos + bin_true_neg) / (bin_true_pos + bin_false_pos + bin_true_neg + bin_false_neg)
    bin_f1 = 2 * (bin_prec * bin_rec) / (bin_prec + bin_rec)



def predict_on_test_set(model, config, test_set):


def main():
    print("Reading in command-line args...")
    args = parse_cmd_args()
    print("Evaluate on test set...")
    results = predict_on_test_set(args.model, args.config, test_set)
    if args.write == True:
        print("Writing to file {}.".format(XYZ))
    print("Done.")
