import os
import argparse
import json
import csv
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch import nn
from utils import get_timestamp


"""
Call:
python3 neural_models.py -c model_configs/config_seq2label_1.json -s
"""


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--location', type=str, help='"local", "midgard" or "rattle". '
                                                           'Indicate which paths will be used.')
    parser.add_argument("-c", "--path_config", type=str, help="Path to hyperparamter/config file (json).")
    parser.add_argument('-n', '--num_threads', type=int, default=10,
                        help='Set the number of threads to use for training by torch.')
    parser.add_argument('-g', '--gpu_num', type=int, default=0, help='The number of the gpu to be used.')
    parser.add_argument('-d', '--device', type=str, help='"cpu" or "cuda". Device to be used.')
    parser.add_argument('-u', '--user', type=str, help='"janis" or "joni". Device to be used.')
    parser.add_argument('-p', '--num_predictions', type=int, default=1000,
                        help='Number of predictions to make for eval on devset.')
    return parser.parse_args()


def load_config(path):
    """Load configuration from json.

    Args:
        path: str
    """
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def adjust_text_len(text, max_len):
    """Multiply char_idxs until max-len.

    Args:
        text: str
        max_len: int
    """
    while len(text) < max_len:
        text += (" " + text)
    text = text[:max_len]
    return text


def get_random_char():
    latin1_chars = ('!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`'
                    'abcdefghijklmnopqrstuvwxyz{|}~\x7f€\x81‚ƒ„…†‡ˆ‰Š‹Œ\x8dŽ\x8f\x90‘’“”•'
                    '–—˜™š›œ\x9džŸ ¡¢£¤¥¦§¨©ª«¬\xad®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓ'
                    'ÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ')
    return random.choice(latin1_chars)


def get_random_english_or_german_token():
    raise NotImplementedError


def noisify_text(text, noise_params):
    """Introduce noise to text by adding english words and repeating/inserting/removing characters.

    Args:
        text: str
        noise_params: dict, subdict of config
    """
    token_threshold = noise_params['token_threshold']
    token_continue_threshold = noise_params['token_continue_threshold']
    char_threshold = noise_params['char_threshold']
    char_continue_threshold = noise_params['char_continue_threshold']
    actions = [1, 2, 3]  # 1: repeat, insert, remove
    # insert noise words
    tokens = text.split(' ')
    noisy_tokens = []
    for i, token in enumerate(tokens):
        num_noise_tokens = 0
        if random.random() > token_threshold:
            noise_token = get_random_english_or_german_token()
            noisy_tokens.append(noise_token)
            num_noise_tokens += 1
            while random.random() > token_continue_threshold and not num_noise_tokens > int(len(tokens) / 2):
                noise_token = get_random_english_or_german_token()
                noisy_tokens.append(noise_token)
                num_noise_tokens += 1
        noisy_tokens.append(token)
    # noise chars
    noisy_token_text = ' '.join(noisy_tokens)
    noisy_chars = ''
    for char in noisy_token_text:
        if random.random() > char_threshold:
            action = random.choice(actions)
            if action == 1:  # repeat
                noise_char = char
                noisy_chars += noise_char
            elif action == 2:  # insert
                noise_char = get_random_char()
                noisy_chars += noise_char
            else:
                continue
            while random.random() > char_continue_threshold:
                noisy_chars += noise_char
        noisy_chars += char
    return noisy_chars


def get_next_batch(csv_reader, batch_size, granularity, char_to_idx, max_length, config):
    """
    Lazy loading of batches. Returns

    Args:
        csv_reader: csv-reader object
        batch_size: int
        granularity: str
        char_to_idx: {str: int}
        max_length: int
        config: dict
    """
    x_list = []
    y_list = []
    if granularity == "binary":
        index = 0 + 3
    elif granularity == "ternary":
        index = 1 + 3
    elif granularity == "finegrained":
        index = 2 + 3
    else:
        raise Exception('ERROR: Granularity level not known.')
    for _ in range(batch_size):
        try:
            row = next(csv_reader)
            adjust_text = adjust_text_len(row[1], max_length)
            if config.get('noisify', False):
                noisy_text = noisify_text(adjust_text)
            else:
                noisy_text = adjust_text
            char_idxs = [char_to_idx.get(char, 'unk') for char in noisy_text]
            label = row[index]
            x_item = np.zeros(max_length)
            for i, idx in enumerate(char_idxs):
                x_item[i] = idx
            x_list.append(x_item)
            y_list.append(int(label))
        except StopIteration:
            break

    x = np.array(x_list, dtype=int)
    y = np.array(y_list, dtype=int)
    return x, y


def get_num_batches(path_train, batch_size):
    """Get the number of batches for one epoch.

    Args:
        path_train: str
        batch_size: int
    """
    with open(path_train, 'r', encoding='utf8') as f:
        num_examples = 0
        for _ in f:
            num_examples += 1
        num_batches = num_examples / batch_size
        if num_examples % batch_size != 0:
            num_batches += 1
        return int(num_batches)


def load_char_to_idx():
    with open('char_to_idx.json', 'r', encoding='utf8') as f:
        return json.load(f)


def create_char_to_idx(path_train):
    """Create a json file mapping characters to idxs.

    Args:
        path_train: str
    """
    i = 0
    char_to_idx = {}
    reader = csv.reader(open(path_train, 'r', encoding='utf8'))
    for row in reader:
        text = row[1]
        for char in text:
            if char in char_to_idx:
                continue
            else:
                char_to_idx[char] = i
                i += 1
    char_to_idx['unk'] = i
    with open('char_to_idx.json', 'w', encoding='utf8') as f:
        json.dump(char_to_idx, f)
    idx_to_char = {val: key for key, val in char_to_idx.items()}
    with open('idx_to_char.json', 'w', encoding='utf8') as f:
        json.dump(idx_to_char, f)


def calc_max_length_trainset(path_train):
    max_len = -1
    with open(path_train, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            len_text = len(row[1])
            if len_text > max_len:
                max_len = len_text
    with open('max_len.json', 'w', encoding='utf8') as f:
        json.dump({'max_len': max_len}, f)


def load_max_len():
    with open('max_len.json', 'r', encoding='utf8') as f:
        return json.load(f)['max_len']


def train_model(config):
    if args.location == 'local':
        path_train = config['path_train']['local']
    elif args.location == 'midgard':
        path_train = config['path_train']['midgard']
    elif args.location == 'rattle':
        path_train = config['path_train']['rattle']
    else:
        raise Exception('Error! Location {} not known.'.format(args.location))
    assert args.user == 'janis' or args.user == 'joni'
    batch_size = config['batch_size']
    granularity = config['granularity']
    num_epochs = config['num_epochs']
    lr = config['learning_rate']
    model_params = config['model_params']
    early_stopping = config.get('early_stopping', False)
    create_char_to_idx(path_train)
    calc_max_length_trainset(path_train)
    print('Load char to idx mapping...')
    char_to_idx = load_char_to_idx()
    print('Load max length...')
    max_length = load_max_len() if 'max_length_text' not in config else config['max_length_text']
    print('Determince device...')
    global device
    if args.device == 'cpu':
        device = 'cpu'
        torch.set_num_threads(args.num_threads)
    elif args.device == 'cuda':
        assert torch.cuda.is_available()
        assert isinstance(args.gpu_num, int)
        device = 'cuda:{}'.format(args.gpu_num)
    else:
        print('No device information in command line arguments. Inferring...')
        device = 'cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    print('Initiate model...')
    model = models[config['model_name']](char_to_idx, **model_params).to(device)
    print('Prepare optimizer and criterion...')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    num_batches = get_num_batches(path_train, batch_size)
    cur_epoch = 0
    cur_batch = 0
    avg_batch_losses = []
    avg_epoch_losses = []
    all_dev_results = []  # list of dicts
    print('Start training...')
    for epoch in range(1, num_epochs + 1):
        print('*** Start epoch [{}/{}] ***'.format(epoch, num_epochs))
        train_reader = csv.reader(open(path_train, 'r', encoding='utf8'))
        losses = []
        cur_epoch = epoch
        if epoch > 5:
            for g in optimizer.param_groups:
                g['lr'] = 0.00003
        elif epoch > 2:
            for g in optimizer.param_groups:
                g['lr'] = 0.0001
        for batch_num in range(num_batches):
            cur_batch = batch_num
            # batch_size = 4 * batch_size if epoch > 2 else batch_size
            x, y = get_next_batch(csv_reader=train_reader, batch_size=batch_size, granularity=granularity,
                                  char_to_idx=char_to_idx, max_length=max_length, config=config)
            if len(x) == 0:
                print('WARNING: Empty batch.')
                continue

            x_train = torch.LongTensor(x).to(device)
            y_train = torch.LongTensor(y).to(device)

            # zero the gradients
            optimizer.zero_grad()

            # propagate forward
            output = model(x_train)

            # compute loss
            loss = criterion(output, y_train)
            losses.append(loss.item())
            # backward propagation
            loss.backward()

            # optimize
            optimizer.step()

            if batch_num % 50 == 0:
                avg_loss = np.mean(losses)
                msg = 'Epoch [{}/{}], batch [{}/{}], avg. loss: {:.4f}'
                print(msg.format(epoch, num_epochs, batch_num, num_batches, avg_loss))
                avg_batch_losses.append(avg_loss)
                losses = []
            if batch_num % 10000 == 0 and batch_num != 0:
                print('Saving current model to disk...')
                save_model(model, config, args.location, cur_epoch, cur_batch, args.user, finale_true=False)

        avg_epoch_losses.append(np.mean(avg_batch_losses))
        avg_batch_losses = []
        print('Avg loss of epoch {}:  {:.4f}'.format(epoch, avg_epoch_losses[-1]))
        print('Predict on devsubset...')
        epoch_results = predict_on_devsubset(model, char_to_idx, max_length, config['path_dev'][args.location], args.num_predictions,
                                             device)
        print('F1-Score: {:.3f}\nAccuracy: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}'.format(
            epoch_results['f1_score'], epoch_results['accuracy'], epoch_results['precision'], epoch_results['recall']))
        all_dev_results.append(epoch_results)
        if early_stopping:
            f1_cur = epoch_results['f1']
            f1_last = all_dev_results[-2]['f1']
            if f1_last >= f1_cur:
                print('EARLY STOPPING! F1 score for this epoch: {:.3f}, last epoch: {:.3f}'.format(f1_cur, f1_last))
                print('STOP TRAINING.')
                return model, cur_epoch, cur_batch, all_dev_results

    return model, cur_epoch, cur_batch, all_dev_results


def get_n_random_examples(path_devset, num_predictions):
    """Extract n random examples from given dataset.

    Args:
        path_devset: str
        num_predictions: int
    """
    len_devset = len([0 for _ in open(path_devset, 'r', encoding='utf8')])
    # draw random examples
    draws = set()
    while len(draws) < num_predictions:
        draws.add(random.randint(0, len_devset))
    # extract drawn examples from file
    drawn_examples = []
    fdev = open(path_devset, 'r', encoding='utf8')
    for i, row in enumerate(csv.reader(fdev)):
        if i in draws:
            drawn_examples.append(row)
    return drawn_examples


def predict_on_devsubset(model, char_to_idx, max_length, path_devset, num_predictions, device):
    """Predict on subset of devset with given model.

    Args:
        model: nn.Model
        char_to_idx: {str: int}
        max_length: int
        path_devset: str
        num_predictions: int, number examples used for prediction
        device: 'cuda:<n>' or 'cpu'
    """
    results = {}
    preds_binary = []
    trues_binary = []
    model.eval()
    drawn_examples = get_n_random_examples(path_devset, num_predictions)
    for i, (text_id, text, masked, label_binary, label_ternary, label_finegrained, source) in enumerate(drawn_examples):
        char_idxs = [char_to_idx.get(char, char_to_idx['unk']) for char in text][:max_length]
        x = np.zeros(max_length)
        for j, idx in enumerate(char_idxs):
            x[j] = idx
        output_raw = model(torch.LongTensor([x]).to(device))
        output = torch.squeeze(output_raw)
        max_prob, prediction = torch.max(output, 0)
        prediction = prediction.item()
        pred_binary = prediction if prediction <= 1 else 1
        preds_binary.append(pred_binary)
        trues_binary.append(float(label_binary))
    model.train()
    results['f1_score'] = f1_score(trues_binary, preds_binary)
    results['accuracy'] = accuracy_score(trues_binary, preds_binary)
    results['recall'] = recall_score(trues_binary, preds_binary)
    results['precision'] = precision_score(trues_binary, preds_binary)
    return results


class SeqToLabelModelConcatAll(nn.Module):

    def __init__(self, char_to_idx, embedding_dim, hidden_gru_size, num_gru_layers, num_classes, dropout, max_len_text, batch_size):
        super(SeqToLabelModelConcatAll, self).__init__()
        self.embedding = nn.Embedding(len(char_to_idx), embedding_dim=embedding_dim)
        self.char_lang_model = nn.GRU(input_size=embedding_dim, hidden_size=hidden_gru_size, dropout=dropout,
                                      num_layers=num_gru_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(max_len_text*hidden_gru_size, num_classes)
        self.batch_size = batch_size if self.training else 1
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embeds = self.embedding(x)
        seq_output, h_n = self.char_lang_model(embeds)
        # all_in_one = torch.reshape(seq_output, (1, -1)) for eval of old models
        all_in_one = torch.reshape(seq_output, (self.batch_size, -1))
        output = self.linear(torch.squeeze(all_in_one))
        return self.logsoftmax(output)


class SeqToLabelModelOutputAndHiddenBi(nn.Module):

    def __init__(self, char_to_idx, embedding_dim, hidden_gru_size, num_gru_layers, num_classes, dropout,
                 batch_size, in_lin_size_fw_h, inbetw_lin_size_fw_h, out_lin_size_fw_h, in_lin_size_bw_h,
                 inbetw_lin_size_bw_h, out_lin_size_bw_h, inbetw_lin_size_final, filter_sizes, strides,
                 num_in_channels, num_out_channels):
        super(SeqToLabelModelOutputAndHiddenBi, self).__init__()
        self.embedding = nn.Embedding(len(char_to_idx), embedding_dim=embedding_dim)
        self.char_lang_model = nn.GRU(input_size=embedding_dim, hidden_size=hidden_gru_size, dropout=dropout,
                                      num_layers=num_gru_layers, batch_first=True, bidirectional=True)
        self.linblock_fw_h = LinBlock(in_lin_size=in_lin_size_fw_h, inbetw_lin_size=inbetw_lin_size_fw_h,
                                      out_lin_size=out_lin_size_fw_h, dropout=dropout)
        self.linblock_bw_h = LinBlock(in_lin_size=in_lin_size_bw_h, inbetw_lin_size=inbetw_lin_size_bw_h,
                                      out_lin_size=out_lin_size_bw_h, dropout=dropout)
        # self.linblock_fw_out = LinBlock(in_lin_size=in_lin_size_fw_out, inbetw_lin_size=inbetw_lin_size_fw_out,
        #                                 out_lin_size=out_lin_size_fw_out, dropout=dropout)
        # self.linblock_bw_out = LinBlock(in_lin_size=in_lin_size_bw_out, inbetw_lin_size=inbetw_lin_size_bw_out,
        #                                 out_lin_size=out_lin_size_bw_out, dropout=dropout)
        self.cnnblock = CNNBlock2(filter_sizes=filter_sizes, embedding_dim=embedding_dim, strides=strides,
                                  num_in_channels=num_in_channels, num_out_channels=num_out_channels)
        # self.linear = nn.Linear(max_len_text*hidden_gru_size, num_classes)
        self.linblock_final = LinBlock(515, inbetw_lin_size=inbetw_lin_size_final,
                                       out_lin_size=num_classes, dropout=0.05)
        self.batch_size = batch_size if self.training else 1
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embeds = self.embedding(x)
        seq_output, h_n = self.char_lang_model(embeds)
        lin_out_fw_h = self.linblock_fw_h(h_n[0])
        lin_out_bw_h = self.linblock_bw_h(h_n[1])
        cnn_out1, cnn_out2, cnn_out3, cnn_out4 = self.cnnblock(seq_output[:, None, :, :])
        cnn_out1_re = torch.reshape(cnn_out1, (self.batch_size, -1))
        cnn_out2_re = torch.reshape(cnn_out2, (self.batch_size, -1))
        cnn_out3_re = torch.reshape(cnn_out3, (self.batch_size, -1))
        cnn_out4_re = torch.reshape(cnn_out4, (self.batch_size, -1))
        feat_vec = torch.cat((lin_out_fw_h, lin_out_bw_h, cnn_out1_re, cnn_out2_re, cnn_out3_re, cnn_out4_re), dim=1)
        output = self.linblock_final(feat_vec)
        out_proba = self.logsoftmax(output)
        return out_proba


class SeqToLabelModelOnlyHiddenBiDeepOriginal(nn.Module):

    def __init__(self, char_to_idx, embedding_dim, hidden_gru_size, num_gru_layers, num_classes, dropout):
        super(SeqToLabelModelOnlyHiddenBiDeepOriginal, self).__init__()
        self.embedding = nn.Embedding(len(char_to_idx), embedding_dim=embedding_dim)
        self.char_lang_model = nn.GRU(input_size=embedding_dim, hidden_size=hidden_gru_size, dropout=dropout,
                                      num_layers=num_gru_layers, batch_first=True, bidirectional=True)
        self.in_lin_size = hidden_gru_size * num_gru_layers
        self.in_betw_size = int(self.in_lin_size / 2)
        self.linblock1 = LinBlock(in_lin_size=self.in_lin_size, inbetw_lin_size=self.in_betw_size,
                                  out_lin_size=50, dropout=dropout)
        self.linblock2 = LinBlock(in_lin_size=self.in_lin_size, inbetw_lin_size=self.in_betw_size,
                                  out_lin_size=50, dropout=dropout)
        self.linear = nn.Linear(100, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embeds = self.embedding(x)
        seq_output, h_n = self.char_lang_model(embeds)
        out1 = self.linblock1(h_n[0])
        out2 = self.linblock2(h_n[0])
        out = self.linear(torch.cat((out1, out2), dim=1))
        out_proba = self.logsoftmax(out)
        return out_proba


class SeqToLabelModelOnlyHiddenBiDeep(nn.Module):

    def __init__(self, char_to_idx, embedding_dim, hidden_gru_size, num_gru_layers, num_classes, dropout):
        super(SeqToLabelModelOnlyHiddenBiDeep, self).__init__()
        self.embedding = nn.Embedding(len(char_to_idx), embedding_dim=embedding_dim)
        self.char_lang_model = nn.GRU(input_size=embedding_dim, hidden_size=hidden_gru_size, dropout=dropout,
                                      num_layers=num_gru_layers, batch_first=True, bidirectional=True)
        self.linblock_in_size = hidden_gru_size
        self.linblock_in_betw_size = hidden_gru_size
        self.linblock_out_size = int(self.linblock_in_betw_size / 2) if self.linblock_in_betw_size > 100 else 50
        self.linblocks = [LinBlock(in_lin_size=self.linblock_in_size, inbetw_lin_size=self.linblock_in_betw_size,
                                   out_lin_size=self.linblock_out_size, dropout=dropout).to(device)
                          for _ in range(num_gru_layers*2)]
        self.linear = nn.Linear(self.linblock_out_size*num_gru_layers*2, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embeds = self.embedding(x)
        seq_output, h_n = self.char_lang_model(embeds)
        lin_outputs = []
        for i, h_n_i in enumerate(h_n):
            lin_outputs.append(self.linblocks[i](h_n_i))
        out = self.linear(torch.cat(lin_outputs, dim=1))
        out_proba = self.logsoftmax(out)
        return out_proba


class SeqToLabelModelOnlyHiddenUniDeep(nn.Module):

    def __init__(self, char_to_idx, embedding_dim, hidden_gru_size, num_gru_layers, num_classes, dropout):
        super(SeqToLabelModelOnlyHiddenUniDeep, self).__init__()
        self.embedding = nn.Embedding(len(char_to_idx), embedding_dim=embedding_dim)
        self.char_lang_model = nn.GRU(input_size=embedding_dim, hidden_size=hidden_gru_size, dropout=dropout,
                                      num_layers=num_gru_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_gru_size*num_gru_layers, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        embeds = self.embedding(x)
        seq_output, h_n = self.char_lang_model(embeds)
        output = self.linear(torch.reshape(h_n, (batch_size, -1)))
        out_proba = self.logsoftmax(output)
        return out_proba


class SeqToLabelModelOnlyHidden(nn.Module):
    """The best model until now was this but without dropout and without early stopping applied!"""

    def __init__(self, char_to_idx, embedding_dim, hidden_gru_size, num_gru_layers, num_classes, dropout):
        super(SeqToLabelModelOnlyHidden, self).__init__()
        self.embedding = nn.Embedding(len(char_to_idx), embedding_dim=embedding_dim)
        self.char_lang_model = nn.GRU(input_size=embedding_dim, hidden_size=hidden_gru_size, dropout=dropout,
                                      num_layers=num_gru_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden_gru_size, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embeds = self.embedding(x)
        seq_output, h_n = self.char_lang_model(embeds)
        output = self.linear(torch.squeeze(h_n))
        out_proba = self.logsoftmax(output)
        return out_proba


class CNNOnly(nn.Module):

    def __init__(self, char_to_idx, embedding_dim, filter_sizes, padding, stride, num_out_channels, inbetw_lin_size,
                 num_classes, dropout, max_len_text, batch_size):
        super(CNNOnly, self).__init__()
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        self.dropout_rt = dropout
        self.max_len_text = max_len_text
        self.batch_size = batch_size
        self.embedding = nn.Embedding(len(char_to_idx), embedding_dim=embedding_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_out_channels, kernel_size=(filter_sizes[0], embedding_dim), stride=stride, padding=padding),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, num_out_channels, kernel_size=(filter_sizes[1], embedding_dim), stride=stride, padding=padding),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, num_out_channels, kernel_size=(filter_sizes[2], embedding_dim), stride=stride, padding=padding),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, num_out_channels, kernel_size=(filter_sizes[3], embedding_dim), stride=stride, padding=padding),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.hier = nn.Sequential(
            nn.Conv1d(1, num_out_channels, kernel_size=20, stride=stride, padding=padding),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # self.linear = nn.Linear(1520, num_classes)
        # self.conv1_out_size = (embedding_dim * max_len_text - filter_sizes[0] + 2 * padding) / (stride) + 1
        # self.conv2_out_size = (embedding_dim * max_len_text - filter_sizes[1] + 2 * padding) / (stride) + 1
        # self.conv3_out_size = (embedding_dim * max_len_text - filter_sizes[2] + 2 * padding) / (stride) + 1
        # self.conv4_out_size = (embedding_dim * max_len_text - filter_sizes[3] + 2 * padding) / (stride) + 1
        #
        # self.conv1_out_flat_size = num_out_channels * self.conv1_out_size
        # self.conv2_out_flat_size = num_out_channels * self.conv2_out_size
        # self.conv3_out_flat_size = num_out_channels * self.conv3_out_size
        # self.conv4_out_flat_size = num_out_channels * self.conv4_out_size
        #
        # self.conv_concat_size = int(self.conv1_out_flat_size + self.conv2_out_flat_size + \
        #                         self.conv3_out_flat_size + self.conv4_out_flat_size)

        self.classifier_layers = nn.Sequential(
            nn.Dropout(self.dropout_rt),
            # nn.Linear(self.conv_concat_size, inbetw_lin_size),
            nn.Linear(396, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rt),
            nn.Linear(256, inbetw_lin_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rt),
            nn.Linear(inbetw_lin_size, num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        embeds = self.embedding(x)
        embeds_add_dim = embeds[:, None, :, :]

        output_conv1 = self.conv1(embeds_add_dim)
        output_conv2 = self.conv2(embeds_add_dim)
        output_conv3 = self.conv3(embeds_add_dim)
        output_conv4 = self.conv4(embeds_add_dim)

        oconv1_re = torch.reshape(output_conv1, (batch_size, -1))
        oconv2_re = torch.reshape(output_conv2, (batch_size, -1))
        oconv3_re = torch.reshape(output_conv3, (batch_size, -1))
        oconv4_re = torch.reshape(output_conv4, (batch_size, -1))

        feat_vec = torch.cat((oconv1_re, oconv2_re, oconv3_re, oconv4_re), dim=1)[:, None, :]

        out_hier1 = self.hier(feat_vec) 
        out_hier1_rs = torch.reshape(out_hier1, (batch_size, -1))

        output = self.classifier_layers(out_hier1_rs)
        # import pdb; pdb.set_trace()
        out_proba = self.logsoftmax(output)
        return out_proba


class CNNHierarch(nn.Module):

    def __init__(self, char_to_idx, embedding_dim, filter_sizes, padding, stride, num_out_channels, inbetw_lin_size_1,
                 inbetw_lin_size_2, out_lin_size, num_classes, dropout, max_len_text, batch_size):
        super(CNNHierarch, self).__init__()
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        self.dropout_rt = dropout
        self.max_len_text = max_len_text
        self.batch_size = batch_size
        self.embedding = nn.Embedding(len(char_to_idx), embedding_dim=embedding_dim)
        self.conv_l1_1 = nn.Sequential(
            nn.Conv1d(1, num_out_channels, kernel_size=filter_sizes[0] * embedding_dim, stride=stride*embedding_dim, padding=(filter_sizes[0]-1)*embedding_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_l1_2 = nn.Sequential(
            nn.Conv1d(1, num_out_channels, kernel_size=filter_sizes[1] * embedding_dim, stride=stride*embedding_dim, padding=(filter_sizes[1]-1)*embedding_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_l1_3 = nn.Sequential(
            nn.Conv1d(1, num_out_channels, kernel_size=filter_sizes[2] * embedding_dim, stride=stride*embedding_dim, padding=(filter_sizes[2]-1)*embedding_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_l1_4 = nn.Sequential(
            nn.Conv1d(1, num_out_channels, kernel_size=filter_sizes[3] * embedding_dim, stride=stride*embedding_dim, padding=(filter_sizes[3]-1)*embedding_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_l2_1 = nn.Sequential(
            nn.Conv1d(num_out_channels, num_out_channels, kernel_size=filter_sizes[0], stride=stride, padding=filter_sizes[0]-1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_l2_2 = nn.Sequential(
            nn.Conv1d(num_out_channels, num_out_channels, kernel_size=filter_sizes[1], stride=stride, padding=filter_sizes[1]-1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_l2_3 = nn.Sequential(
            nn.Conv1d(num_out_channels, num_out_channels, kernel_size=filter_sizes[2], stride=stride, padding=filter_sizes[2]-1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_l2_4 = nn.Sequential(
            nn.Conv1d(num_out_channels, num_out_channels, kernel_size=filter_sizes[3], stride=stride, padding=filter_sizes[3]-1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # self.lin_layers_1 = nn.Sequential(
        #     nn.Dropout(self.dropout_rt),
        #     # nn.Linear(self.conv_concat_size, inbetw_lin_size),
        #     nn.Linear(4040, inbetw_lin_size_1),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(self.dropout_rt),
        #     nn.Linear(inbetw_lin_size_1, out_lin_size),
        # )
        #
        # self.lin_layers_2 = nn.Sequential(
        #     nn.Dropout(self.dropout_rt),
        #     # nn.Linear(self.conv_concat_size, inbetw_lin_size),
        #     nn.Linear(4040, inbetw_lin_size_2),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(self.dropout_rt),
        #     nn.Linear(inbetw_lin_size_2, num_classes),
        # )

        self.lin_layers_per_filter_1 = nn.Sequential(
            nn.Dropout(self.dropout_rt),
            nn.Linear(300, 100),
            nn.ReLU(inplace=True),
        )

        self.lin_layers_per_filter_2 = nn.Sequential(
            nn.Dropout(self.dropout_rt),
            nn.Linear(304, 100),
            nn.ReLU(inplace=True),
        )

        self.lin_layers_per_filter_3 = nn.Sequential(
            nn.Dropout(self.dropout_rt),
            nn.Linear(308, 100),
            nn.ReLU(inplace=True),
        )

        self.lin_layers_per_filter_4 = nn.Sequential(
            nn.Dropout(self.dropout_rt),
            nn.Linear(312, 100),
            nn.ReLU(inplace=True),
        )

        self.final_layer = nn.Sequential(
            nn.Dropout(self.dropout_rt),
            nn.Linear(400, num_classes),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        embeds = self.embedding(x)
        embeds_flat = torch.reshape(embeds, (batch_size, -1))
        embeds_add_dim = embeds_flat[:, None, :]

        output_conv1 = self.conv_l1_1(embeds_add_dim)
        output_conv2 = self.conv_l1_2(embeds_add_dim)
        output_conv3 = self.conv_l1_3(embeds_add_dim)
        output_conv4 = self.conv_l1_4(embeds_add_dim)

        output_conv_l2_1 = self.conv_l2_1(output_conv1)
        output_conv_l2_2 = self.conv_l2_2(output_conv2)
        output_conv_l2_3 = self.conv_l2_3(output_conv3)
        output_conv_l2_4 = self.conv_l2_4(output_conv4)

        oconv1_re2 = torch.reshape(output_conv_l2_1, (batch_size, -1))
        oconv2_re2 = torch.reshape(output_conv_l2_2, (batch_size, -1))
        oconv3_re2 = torch.reshape(output_conv_l2_3, (batch_size, -1))
        oconv4_re2 = torch.reshape(output_conv_l2_4, (batch_size, -1))

        lin_out1 = self.lin_layers_per_filter_1(oconv1_re2)
        lin_out2 = self.lin_layers_per_filter_2(oconv2_re2)
        lin_out3 = self.lin_layers_per_filter_3(oconv3_re2)
        lin_out4 = self.lin_layers_per_filter_4(oconv4_re2)

        final_feat_vec = torch.cat((lin_out1, lin_out2, lin_out3, lin_out4), dim=1)

        output = self.final_layer(final_feat_vec)
        out_proba = self.logsoftmax(output)
        return out_proba


def save_model(trained_model, config, location, num_epochs, num_batches, user, all_dev_results=None,
               finale_true=False):
    if location == 'local':
        path_out = 'models'
    elif location == 'midgard':
        path_out = '/home/user/jgoldz/storage/shared_task/models'
    elif location == 'rattle':
        if user == "janis":
            path_out = '/srv/scratch3/jgoldz_jschab/shared_task/models'
        elif user == "joni":
            path_out = '/srv/scratch3/jschab_jgoldz/shared_task/models'
    else:
        raise Exception('Error! Location "{}" not known.'.format(location))
    time_stamp = get_timestamp()
    fname_model = '{model_name}_{config_id}_{num_epochs}_{num_batches}_{timestamp}_end{finale_true}.model'.format(
        model_name=config['model_name'], config_id=config['config_id'],
        num_epochs=num_epochs, num_batches=num_batches,
        timestamp=time_stamp, finale_true=finale_true)
    fpath_model = os.path.join(path_out, fname_model)
    torch.save(trained_model, fpath_model)
    print('Model saved to {}'.format(fpath_model))
    if all_dev_results:
        fname_dev_results = '{model_name}_{config_id}_{num_epochs}_{num_batches}_{timestamp}_end{finale_true}.results'.\
            format(model_name=config['model_name'], config_id=config['config_id'],
            num_epochs=num_epochs, num_batches=num_batches,
            timestamp=time_stamp, finale_true=finale_true)
        fpath_dev_results = os.path.join(path_out, fname_dev_results)
        with open(fpath_dev_results, 'w', encoding='utf8') as f:
            for epoch_result in all_dev_results:
                f.write(str(epoch_result) + '\n')
        print('Dev-results saved to {}'.format(fpath_dev_results))


class CNNBlock(nn.Module):

    def __init__(self, filter_sizes, embedding_dim, stride, num_in_channels, num_out_channels, dropout):
        super(CNNBlock, self).__init__()
        self.filter_sizes = filter_sizes
        self.dropout_rt = dropout
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_in_channels, num_out_channels, kernel_size=filter_sizes[0] * embedding_dim,
                      stride=stride * embedding_dim, padding=(filter_sizes[0] - 1) * embedding_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_in_channels, num_out_channels, kernel_size=filter_sizes[1] * embedding_dim,
                      stride=stride * embedding_dim, padding=(filter_sizes[1] - 1) * embedding_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_in_channels, num_out_channels, kernel_size=filter_sizes[2] * embedding_dim,
                      stride=stride * embedding_dim, padding=(filter_sizes[2] - 1) * embedding_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(num_in_channels, num_out_channels, kernel_size=filter_sizes[3] * embedding_dim,
                      stride=stride * embedding_dim, padding=(filter_sizes[3] - 1) * embedding_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        output_conv1 = self.conv1(x)
        output_conv2 = self.conv2(x)
        output_conv3 = self.conv3(x)
        output_conv4 = self.conv4(x)
        return output_conv1, output_conv2, output_conv3, output_conv4


class CNNBlock2(nn.Module):

    def __init__(self, filter_sizes, embedding_dim, strides, num_in_channels, num_out_channels):
        super(CNNBlock2, self).__init__()
        self.filter_dim_2 = int(embedding_dim / 5)
        self.stride_dim_2 = self.filter_dim_2
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels, kernel_size=(filter_sizes[0], self.filter_dim_2),
                      stride=(strides[0], self.stride_dim_2), padding=(0, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels, kernel_size=(filter_sizes[1], self.filter_dim_2),
                      stride=(strides[1], self.stride_dim_2), padding=(0, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels, kernel_size=(filter_sizes[1], self.filter_dim_2),
                      stride=(strides[2], self.stride_dim_2), padding=(0, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels, kernel_size=(filter_sizes[1], self.filter_dim_2),
                      stride=(strides[3], self.stride_dim_2), padding=(0, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        output_conv1 = self.conv1(x)
        output_conv2 = self.conv2(x)
        output_conv3 = self.conv3(x)
        output_conv4 = self.conv4(x)
        return output_conv1, output_conv2, output_conv3, output_conv4


class LinBlock(nn.Module):

    def __init__(self, in_lin_size, inbetw_lin_size, out_lin_size, dropout):
        super(LinBlock, self).__init__()
        self.dropout_rt = dropout

        self.lin_layers_1 = nn.Sequential(
            nn.Dropout(self.dropout_rt),
            nn.Linear(in_lin_size, inbetw_lin_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rt),
            nn.Linear(inbetw_lin_size, out_lin_size),
        )

    def forward(self, x):
        return self.lin_layers_1(x)


class GRUCNN(nn.Module):

    def __init__(self, char_to_idx, embedding_dim, hidden_gru_size, num_gru_layers, num_classes, dropout, max_len_text,
                 batch_size, filter_sizes, padding, stride, num_in_channels, num_out_channels, in_lin_size,
                 inbetw_lin_size):
        super(GRUCNN, self).__init__()
        self.batch_size = batch_size if self.training else 1
        self.filter_sizes_dim_2 = 4*[hidden_gru_size * 2 * num_gru_layers]
        self.embedding = nn.Embedding(len(char_to_idx), embedding_dim=embedding_dim)
        self.char_lang_model = nn.GRU(input_size=embedding_dim, hidden_size=hidden_gru_size, dropout=dropout,
                                      num_layers=num_gru_layers, batch_first=True, bidirectional=False)
        self.cnn_block = CNNBlock(filter_sizes=filter_sizes, stride=stride, num_in_channels=num_in_channels,
                                  num_out_channels=num_out_channels, dropout=dropout, embedding_dim=embedding_dim)
        self.lin_block = LinBlock(in_lin_size=in_lin_size, inbetw_lin_size=inbetw_lin_size,
                                  out_lin_size=num_classes, dropout=dropout)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        embeds = self.embedding(x)
        seq_output, h_n = self.char_lang_model(embeds)
        hn_re = torch.reshape(h_n, (batch_size, -1))[:, None, :]
        all_in_one = torch.cat((seq_output, hn_re), dim=1)[:, None, :, :]
        all_in_one_flat = torch.reshape(all_in_one, (batch_size, 1, -1))
        cnn_out1, cnn_out2, cnn_out3, cnn_out4 = self.cnn_block(all_in_one_flat)

        cnn_out_flat1 = torch.reshape(cnn_out1, (batch_size, -1))
        cnn_out_flat2 = torch.reshape(cnn_out2, (batch_size, -1))
        cnn_out_flat3 = torch.reshape(cnn_out3, (batch_size, -1))
        cnn_out_flat4 = torch.reshape(cnn_out4, (batch_size, -1))

        feat_vec = torch.cat((cnn_out_flat1, cnn_out_flat2, cnn_out_flat3, cnn_out_flat4), dim=1)

        output = self.lin_block(feat_vec)
        out_proba = self.logsoftmax(output)
        return out_proba


class TransformerLin(nn.Module):

    def __init__(self, char_to_idx, embedding_dim, hidden_gru_size, num_gru_layers, num_classes, dropout, max_len_text,
                 batch_size, filter_sizes, padding, stride, num_in_channels, num_out_channels, in_lin_size,
                 inbetw_lin_size):
        super(GRUCNN, self).__init__()
        self.batch_size = batch_size if self.training else 1
        self.transformer = self.load_pretrained(path_transformer)
        self.lin_block = LinBlock(in_lin_size=in_lin_size, inbetw_lin_size=inbetw_lin_size,
                                  out_lin_size=num_classes, dropout=dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        transformer_out = self.transformer(x)
        output = self.lin_block(transformer_out)
        return output


args = None


models = {
    'SeqToLabelModelOnlyHidden': SeqToLabelModelOnlyHidden,
    'SeqToLabelModelConcatAll': SeqToLabelModelConcatAll,
    'SeqToLabelModelOnlyHiddenBiDeep': SeqToLabelModelOnlyHiddenBiDeep,
    'CNNOnly': CNNOnly,
    'CNNHierarch': CNNHierarch,
    'GRUCNN': GRUCNN,
    'SeqToLabelModelOnlyHiddenUniDeep': SeqToLabelModelOnlyHiddenUniDeep,
    'SeqToLabelModelOnlyHiddenBiDeepOriginal': SeqToLabelModelOnlyHiddenBiDeepOriginal,
    'SeqToLabelModelOutputAndHiddenBi': SeqToLabelModelOutputAndHiddenBi
}


def main():
    global args
    print('Parse cmd line args...')
    args = parse_cmd_args()
    print('Loading config from {}...'.format(args.path_config))
    config = load_config(args.path_config)
    print('Initiate training procedure...')
    trained_model, num_epochs, num_batches, all_dev_results = train_model(config)
    print('Saving trained model...')
    save_model(trained_model, config, args.location, num_epochs, num_batches, args.user, all_dev_results,
               finale_true=True)


if __name__ == '__main__':
    main()
