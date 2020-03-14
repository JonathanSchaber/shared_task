import csv
import datetime
import re


def get_timestamp():
    timestamp = datetime.datetime.now().strftime('%c')
    return re.sub(r'\s+', r'_', timestamp)


def count_num_classes(path_in, idx):
    """Count the number of classes.

    Args:
        path_in: str, path to csv
        idx: int, index of row where class label is
    """
    labels = set()
    with open(path_in, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[idx]
            labels.add(label)
    return len(labels)
