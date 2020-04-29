import csv 


def parse_cmd_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str)
    parser.add_argument("-o", "--outfile", type=str)
    return parser.parse_args()

def read_and_write(infile, outfile):
    csv_reader= csv.reader(open(infile))
    csv_writer = csv.writer(open(outfile))

    for row in reader:
        text_id, label_binary, label_ternary, label_finegrained, pred_binary, pred_ternary, pred_finegrained, text, masked, source = line
        crv_writer.writerow([text_id, "gsw" if pred_binary == 0 else "not_gsw", "0.9"])

def main():
    args = parse_cmd_args()
    read_and_write(args.infile, args.outfile)
    
if __name__ == "__main__":
    main()
