import re
import os
import argparse
import csv
import emoji

# Some Variables (Lists of characters, regexes etc.)

swiss_chars = [char for char in "qQwWeRtTzZuUiIoOpPüèÜaAsSdDfFgGhHjJkKlLöéÖäàÄyYxXcCvVbBnNmMçÇ"]

punctuation = re.compile(r"[<>,;.:?!'¿^`´+\-\\*%&/()=0123456789]")

masks = re.compile(r"[MASK_MENTION|MASK_HASHTAG|MASK_URL]")

emojis = emoji.get_emoji_regexp()


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


def check_sentences(text, threshold=80, print_only=False):
    """check if a given text consists of more non-swiss chars than allowed by the threshold

    Args:
        text: str
        threshold: int
        print_only: boolean
    Returns:
        True (if below threshold)
        False XOR str (if print_only == True)
    """
    non_white_text = re.sub(masks, "", re.sub(emojis, "", re.sub(punctuation, "", re.sub("\s", "", text))))
    num_chars = len(non_white_text)
    num_non_swiss_chars = 0
    for char in non_white_text:
        if char not in swiss_chars:
            num_non_swiss_chars += 1
    ratio = num_non_swiss_chars / num_chars * 100 if num_chars != 0 else 0
    if ratio > threshold:
        return "POSSIBLE NON_SWISS GERMAN TEXT:" + text if print_only else False
    else:
        return text

def file_exists_check(path):
    """checks if outfile already exists. if yes, user is asked if it should be overwritten.

    Args:
        path: str
    Returns:
        True (if file doesn't exist XOR user tells to overwrite
        False (else)
    """
    if os.path.exists(path):
        overwrite = input("Outfile already existing. Overwrite [Y/n]?: ")
        if overwrite == "Y":
            os.system("rm {}".format(path))
            return True
        else:
            print("\nAborted.")
            return False
    else:
        return True


def process_file(path_in, path_out, threshold):
    """Process infile line by line. If text is below threshold it is written to outfile.

    Args:
        path_in: str
        path_out: str
        threshold: int
    Returns:
        None
    """
    infile = open(path_in, "r", encoding="utf-8")
    outfile = open(path_out, "w", encoding="utf-8")
    csv_reader = csv.reader(infile)
    csv_writer = csv.writer(outfile)
    for i, line in enumerate(csv_reader):
        try:
            text_id, text, masked, label_binary, label_ternary, label_finegrained, source = line
        except ValueError:
            if line == ['Place for parser output']:
                pass
            else:
                import pdb; pdb.set_trace()
        if print_only:
            print(check_sentences(text, threshold, print_only))
        else:
            # return False
            swiss_text = check_sentences(text, threshold)
            if i % 10000 == 0:
                print("Processed line #{}".format(i) + " {}".format(text))
            if swiss_text:
                    csv_writer.writerow([text_id, text, masked, label_binary, label_ternary, label_finegrained, source])
    infile.close()
    outfile.close()



def main():
    args = parse_cmd_args()
    path_in = args.path_in
    path_out = args.path_out
    threshold = args.threshold
    global print_only
    print_only = True if args.print_only else False
    print("Processing the follwoing file: {}".format(path_in))
    if print_only:
        process_file(path_in, path_out, threshold)
    else: 
        file_exists_check(path_out)
        process_file(path_in, path_out, threshold)


if __name__ == "__main__":
    main()
