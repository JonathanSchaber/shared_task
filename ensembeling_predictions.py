import argparse
import csv

def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', type=str, action='append', nargs='+', help='path to prediction files')
    parser.add_argument('-o', '--output_file', type=str, help='path to prediction files')
    return parser.parse_args()


def majority_decision(list_files, path_output_file):
    """Read in the different predictions, compute definitive output file and write to file.

    Args:
        list_files: list
        outfile: str
    Return:
        None
    """
    num_files = len(list_files)
    definite_preds = []

    for i in num_files:

    


def main():
    print('Parse cmd line args...')
    args = parse_cmd_args()
    list_files = args.input_files[0]
    output_file = args.output_file
    if len(list_files) % 2 != 0:
        print("Warning: Even number of files provided!")
    print("Calculating majority decisions...")
    maj_preds = majority_decision(list_files)


if __name__ == '__main__':
    main()
