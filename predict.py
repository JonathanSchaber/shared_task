import csv
import json
import numpy as np
import torch
import torch.nn
from neural_models import *
# from bigram_based_models import *


"""Script to let given model predict on given data.

Example call: 
python3 predict.py -m <path_to_model> -t <torch or sklearn> -i <path_input_data> -o <path_output_file>
python3 predict.py -m /home/user/jgoldz/storage/shared_task/models/SeqToLabelModelOnlyHidden_seq2label_binary_1_1_30_Wed_Mar_11_21:55:12_2020.model -t torch -i /home/user/jgoldz/storage/shared_task/data/main/dev_main.csv -o testpred.csv -c /home/user/jgoldz/shared_task/model_configs/config_seq2label_1.json

python3 predict.py -m /home/user/jgoldz/storage/shared_task/models/CNNOnly_CNNOnly_binary_1/CNNOnly_CNNOnly_binary_1_2_55540_Sun_Mar_15_00:35:35_2020_endTrue.model -t torch -i /home/user/jgoldz/storage/shared_task/data/main/dev_main.csv -o /home/user/jgoldz/storage/shared_task/models/CNNOnly_CNNOnly_binary_1/predict_on_entire_devset.csv -c model_configs/config_cnnonly_1.json
"""


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--path_model', type=str, help='Path to trained model.')
    parser.add_argument('-t', '--type', type=str, help='Either "torch" or "sklearn".')
    # parser.add_argument('-l', '--location', type=str, help='Either "local", "midgard" or "rattle"')
    parser.add_argument('-i', '--path_in', type=str, help='Path to input file.')
    parser.add_argument('-o', '--path_out', type=str, help='Path to output file.')
    parser.add_argument('-c', '--path_config', type=str, help='Path to config file.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='Number of the gpu to be used.')
    return parser.parse_args()


def load_model(path_model, model_type, device):
    """Load the model from disk.

    Args:
        path_model: str
        model_type: str
        device: str
    """
    if model_type == 'torch':
        model = torch.load(path_model).to(device)
        if hasattr(model, 'linblocks'):
            for linblock in model.linblocks:
                linblock.to(device)
        model.eval()
        return model
    elif model_type == 'sklearn':
        raise NotImplementedError
    else:
        raise Exception('Model type not known.')


def get_num_examples(path_in):
    """Count the number of examples in the input data.

    Args:
        path_in: str
    """
    i = 0
    with open(path_in, 'r', encoding='utf8') as f:
        for _ in f:
            i += 1
    return i


def predict_on_input(model, model_type, path_in, config, max_examples, device):
    """Make prediction on the input data with the given model.

    Args:
        model: either torch.nn.Model or sklearn-model
        model_type: str
        path_in: str
        config: dict
        max_examples: int
        device: str
    """
    char_to_idx = load_char_to_idx()
    max_length = load_max_len() if 'max_length_text' not in config else config['max_length_text']
    if not max_examples:
        max_examples = get_num_examples(path_in)
    predictions = []
    if model_type == 'torch':
        reader = csv.reader(open(path_in, 'r', encoding='utf8'))
        for i, row in enumerate(reader):
            text_id, text, masked, label_binary, label_ternary, label_finegrained, source = row
            adjust_text = adjust_text_len(text, max_length, config)
            text_idxs = [char_to_idx.get(char, char_to_idx['unk']) for char in adjust_text]
            x = np.zeros(max_length)
            for j, idx in enumerate(text_idxs):
                x[j] = idx
            output_raw = model(torch.LongTensor([x]).to(device))
            output = torch.squeeze(output_raw)
            max_prob, prediction = torch.max(output, 0)
            pred_binary = prediction.item() if prediction.item() <= 1 else 1
            if config['granularity'] != 'binary':
                pred_ternary = prediction.item() if prediction.item() <= 2 else 2
                if config['granularity'] == 'finegrained':
                    pred_finegrained = prediction.item()
                else:
                    pred_finegrained = 'NULL'
            else:
                pred_ternary = 'NULL'
                pred_finegrained = 'NULL'
            predictions.append((text_id, label_binary, label_ternary, label_finegrained, pred_binary,
                                pred_ternary, pred_finegrained, text, masked, source))
            if i == max_examples - 1:
                print('Predicted on example [{}/{}]'.format(i, max_examples))
                break
            else:
                print('Predicted on example [{}/{}]\r'.format(i, max_examples), end='\r')
    return predictions


def write_preds_to_file(predictions, path_out):
    with open(path_out, 'w', encoding='utf8') as f:
        writer = csv.writer(f)
        for row in predictions:
            writer.writerow(row)


def load_config(path_config):
    """Load the config.

    Args:
        path_config: str
    """
    with open(path_config, 'r', encoding='utf8') as f:
        return json.load(f)


def main():
    print('Parsing command line args...')
    args = parse_cmd_args()
    print('Loading config...')
    config = load_config(args.path_config)
    print('Loading model...')
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    model = load_model(args.path_model, args.type, device)
    print('Make predictions...')
    predictions = predict_on_input(model, args.type, args.path_in, config, None, device)
    print('Write Predictions to file...')
    write_preds_to_file(predictions, args.path_out)


if __name__ == '__main__':
    main()
