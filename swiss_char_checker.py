import re
import argparse
import csv

swiss_chars = [char for char in "qQwWeRtTzZuUiIoOpPüèÜaAsSdDfFgGhHjJkKlLöéÖäàÄyYxXcCvVbBnNmMçÇ"]

punctuation = re.compile(r"[<>,;.:?'¿^`´+\-\\*%&/()=0123456789]")


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--path_in", type=str, default="data/main/main.csv", help="Path to main.csv")
    parser.add_argument("-o", "--path_out", type=str, default="data/main/main_swiss_char.csv",
                        help="Path to output file")
    parser.add_argument("-t", "--threshold", type=int, default=80, help="threshold of non-swiss-characters to predict text as non-swiss-german")
    parser.add_argument("-p", "--print_only", default=False, action="store_true",
                        help="Do not write to file - only print non-swiss-char tweets.")

    return parser.parse_args()


def check_sentences(text, threshold):
    non_white_text = re.sub(punctuation, "", re.sub("\s", "", text))
    num_chars = len(non_white_text)
    num_non_swiss_chars = 0
    for char in non_white_text:
        if char not in swiss_chars:
            num_non_swiss_chars += 1
    ratio = num_non_swiss_chars / num_chars * 100 if num_chars != 0 else 0
    if ratio > threshold:
        return text
    else:
        return "POSSIBLE SWISS GERMAN TEXT FOUND: " + text

def process_file(path_in, path_out, threshold):
    with open(path_in, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            try:
                text_id, text, masked, label, source = line
            except ValueError:
                if line == ['Place for parser output']:
                    pass
                else:
                    import pdb; pdb.set_trace()
            if print_only:
                print(check_sentences(text, threshold))
            else:
                return False


def main():
    args = parse_cmd_args()
    path_in = args.path_in
    path_out = args.path_out
    threshold = args.threshold
    global print_only
    print_only = True if args.print_only else False
    print("Processing the follwoing file: {}".format(path_in))
    process_file(path_in, path_out, threshold)

    #check_sentences("hAlloo=====000000m i bims :-) ahah, wie ga@sg@sg@sg@sg@sg@sg@sg@sg@sgaaaååats?", threshold)
    # check_sentences("data/main/main.csv", 80)


if __name__ == "__main__":
    main()
