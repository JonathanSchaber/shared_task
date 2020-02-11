import csv
import json


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
    pass


def get_top_n(bigram_counts):
    """Only return the n most frequent bigrams.

    Args:
        bigram_counts: {bigram: num occurences}
    Return:
        {bigram: num occurences}
    """
    pass


def get_bigram_to_dim_mapping():
    pass


def dump_bigram_to_dim_mapping():
    pass


def main():
    path_in = 'data/main/main.csv'
    bigram_counts_ch, bigram_counts_other = count_bigrams(path_in)
    top_n_ch = get_top_n(bigram_counts_ch)
    top_n_other = get_top_n(bigram_counts_other)
    bigram_to_dim_mapping = get_bigram_to_dim_mapping(top_n_ch, top_n_other)
    dump_bigram_to_dim_mapping(bigram_to_dim_mapping)





if __name__ == '__main__':
    main()