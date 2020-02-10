import csv

"""
p = NoahParser(path_in, path_out)
p.copy_to_main_file()


"""


class Parser:
    
    def __init__(self, path_in):
        self.path_in = path_in
        self.infile = open(self.path_in, 'r', encoding='utf8')

    def copy_to_main_file(self):
        """Copy the loaded file to the main file."""
        raise NotImplementedError


class NoahParser(Parser): 
    pass


class ChatmaniaParser(Parser):
    pass


class SwissCrawlParser(Parser):

    name = 'swisscrawl'
    num_lines = 562525
    out_path = 'data/main/swisscrawl_parsed.csv'

    def copy_to_main_file(self):
        """Copy the loaded file to the main file."""
        reader = csv.reader(self.infile)
        writer = csv.writer(open(self.out_path, 'w', encoding='utf8'))
        for i, row in enumerate(reader):
            if i == 0:
                continue
            writer.writerow([row[0], '1', self.name])
            if i % 10000:
                print('Processed line {} of {}.'.format(i+1, self.num_lines))


def main():
    path_in = 'data/swisscrawl/swisscrawl-2019-11-23.csv'
    p = SwissCrawlParser(path_in)
    p.copy_to_main_file()


if __name__ == '__main__':
    main()
