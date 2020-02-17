import os
import csv
import json
import argparse
from collections import defaultdict
import numpy as np


"""Generate a bigram representation of the corpus.

To use default paths use: python3 generate_bigram_repr.py

To manually set input and output path use: python3 generate_bigram_repr.py -i <path_in> -o <path_out>

If this script has not been executed before or a new bigram-to-dimension-mapping should be created, use:
python3 generate_bigram_repr.py -g


"""


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--path_in", type=str, default='data/main/main.csv', help="Path to main.csv")
    parser.add_argument("-o", "--path_out", type=str, default='data/main/main_bigr_repr.csv',
                        help="Path to output file")
    parser.add_argument('-g', '--gen_mapping', default=False, action='store_true',
                        help='Generate new dictionary for bigram to dimension mapping.')
    parser.add_argument('-c', '--ch', type=int, default=2000, help='The top n bigrams are used as ch features.')
    parser.add_argument('-w', '--world', type=int, default=10000, help='The top n bigrams are used as features '
                                                                       'for other languages.')
    return parser.parse_args()


def count_bigrams(path_in, threshold=None):
    """Count bigrams for ch and other.

    Args:
        path_in: str, path to input csv file
        threshold: None or int, if int, bigrams occurring
            less than int are not in returned dict.
    Return:
        Tuple containing:
            bigram_counts_ch: {bigram: num occurences}
            bigram_counts_other: {bigram: num occurences}
    """
    bigram_counts_ch = defaultdict(int)
    bigram_counts_other = defaultdict(int)
    with open(path_in, 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            try:
                text_id, text, masked, label, source = line
            except ValueError:
                if line == ['Place for parser output']:
                    pass
                else:
                    import pdb; pdb.set_trace()
            cur_dict = bigram_counts_ch if label == '1' else bigram_counts_other
            bigrams = [bi for bi in zip(text, text[1:])]
            for bigram in bigrams:
                cur_dict[bigram[0]+bigram[1]] += 1
    return bigram_counts_ch, bigram_counts_other


def get_top_n(bigram_counts, n):
    """Only return the n most frequent bigrams.

    Args:
        bigram_counts: {bigram: num occurences}
        n: int
    Return:
        {bigram: num occurences}
    """
    counts = sorted(bigram_counts.values(), reverse=True)
    threshold = counts[n-1]
    return {bigram: count for bigram, count in bigram_counts.items() if count >= threshold}


def get_bigram_to_dim_mapping(top_n_ch, top_n_other):
    """Produce Dictionary that maps bigrams to dimensions

    :param top_n_ch: {bigram: num occurences}
    :param top_n_other: {bigram: num occurences}
    :return: {bigram<str>: dim<int>}
    """
    mapping = {}
    i = 0
    all_bigrams = set(top_n_ch.keys()).union(set(top_n_other.keys()))
    for bigram in all_bigrams:
        mapping[bigram] = i
        i += 1

    return mapping


def dump_bigram_to_dim_mapping(bigram_to_dim_mapping):
    """Dump bigram-to-dim-mapping into json.

    :param bigram_to_dim_mapping: {bigram<str>: dim<int>}
    """
    with open('bigram_to_dim_mapping.json', 'w', encoding='utf8') as f:
        json.dump(bigram_to_dim_mapping, f)


def get_num_lines(path):
    """Count the number of lines in a file.

    :param path: str
    :return: int
    """
    count = 0
    with open(path, 'r', encoding='utf8') as f:
        for _ in f:
            count += 1
    return count


def gen_bigram_repr(path_in, path_out):
    """Generate a bigram representation of the input corpus.

    :param path_in: path to input csv-file.
    :param path_out: path to output csv-file.
    """
    path_mapping = 'bigram_to_dim_mapping.json'
    if not os.path.exists(path_mapping):
        raise Exception('Bigram-to-dim-mapping not found. Please generate the mapping first!')

    with open(path_mapping, 'r', encoding='utf8') as f:
        mapping = json.load(f)
    num_dims = len(mapping)

    fin = open(path_in, 'r', encoding='utf8')
    fout = open(path_out, 'w', encoding='utf8')
    csv_reader = csv.reader(fin)
    csv_writer = csv.writer(fout)
    for line in csv_reader:
        try:
            text_id, text, masked, label, source = line
        except ValueError:
            if line == ['Place for parser output']:
                continue
            else:
                import pdb; pdb.set_trace()
        ohe = np.zeros(num_dims)
        bigrams = [''.join(bi) for bi in zip(text, text[1:])]
        for bigram in bigrams:
            if bigram in mapping:
                ohe[mapping[bigram]] += 1
        csv_writer.writerow([text_id, ', '.join([str(value) for value in ohe])])
        if int(text_id) % 10000 == 0:
            print("Processed line {}.".format(text_id))




def main():
    args = parse_cmd_args()
    if args.gen_mapping:
        bigram_counts_ch, bigram_counts_other = count_bigrams(args.path_in)
        top_n_ch = get_top_n(bigram_counts_ch, args.ch)
        top_n_other = get_top_n(bigram_counts_other, args.world)
        bigram_to_dim_mapping = get_bigram_to_dim_mapping(top_n_ch, top_n_other)
        dump_bigram_to_dim_mapping(bigram_to_dim_mapping)

    gen_bigram_repr(args.path_in, args.path_out)


if __name__ == '__main__':
    main()
