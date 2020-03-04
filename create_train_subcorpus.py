import csv
import argparse


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--path_in", type=str, help="Path to whole train data.")
    parser.add_argument("-g", "--granularity", type=str, help="Specify granularity (binary, ternary, or finegrained).")
    parser.add_argument("-n", "--number", type=int, help="Number of examples per label. Default is 1000.")
    return parser.parse_args()


def create_sub_train_corpus(path_in, granularity, number=1000):
    """Compute Subcorpus and write to file
    Args:
        path_in: str
        granularity: str
        number: int
    Returns:
        None
    """
    lines_to_write = []
    if granularity == "binary":
        labels_dict = {
            0:0,
            1:0
            }
    elif granularity == "ternary":
        labels_dict = {
            0:0,
            1:0,
            2:0
            }
    elif granularity == "finegrained":
        labels_dict = {
	    0:0,
	    1:0,
	    2:0,
	    3:0,
	    4:0,
	    5:0,
	    6:0,
	    7:0,
	    8:0,
	    9:0,
	    10:0,
	    11:0,
	    12:0,
	    13:0,
	    14:0,
	    15:0
	  }
    else:
        raise Exception("ERROR: granularity unknown!")

    with open(path_in, "r",  encoding="utf8") as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            if not all(counts == number for counts in labels_dict.values()):
                label_relevant = line[{"binary":1, "ternary":2, "finegrained":3}[granularity]]
                if labels_dict[int(label_relevant.strip())] >= number:
                    continue
                else:
                    lines_to_write.append(line)
                    labels_dict[int(label_relevant.strip())] += 1
            else:
                break
    with open(path_in.rstrip(".csv") + "_" + granularity + "_" + str(number) + ".csv", "w", encoding="utf8") as f:
        csv_writer = csv.writer(f)
        for line in lines_to_write:
            csv_writer.writerow(line)



def main():
    print('Parse cmd args...')
    args = parse_cmd_args()
    print("Creating Sub-Corpus with granularity {} and {} examples per language.".format(args.granularity, str(args.number)))
    create_sub_train_corpus(args.path_in, args.granularity, args.number) 

if __name__ == "__main__":
    main()

