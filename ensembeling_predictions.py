import argparse

def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', type=str, action='append', nargs='+', help='path to prediction files')
    parser.add_argument('-p', '--num_predictions', type=int, default=1000,
                        help='Number of predictions to make for eval on devset.')
    return parser.parse_args()
