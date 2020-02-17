import os
import re
import csv
import json
import xml.etree.ElementTree as ET


"""
p = NoahParser(path_in, path_out)
p.copy_to_main_file()


"""


# *****************************
# ********** Cleaners *********
# *****************************


class Cleaner:
    mask_dict = {
        'MASK_URL': re.compile((r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org'
                           r'|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|'
                           r'post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au'
                           r'|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|'
                           r'ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk'
                           r'|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|'
                           r'gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il'
                           r'|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la'
                           r'|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq'
                           r'|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa'
                           r'|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd'
                           r'|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg'
                           r'|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg'
                           r'|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\('
                           r'[^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*'
                           r'?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:'
                           r"[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|"
                           r"info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|"
                           r"ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn"
                           r"|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu"
                           r"|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|"
                           r"fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|"
                           r"hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|"
                           r"kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|"
                           r"mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|"
                           r"no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|"
                           r"rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|"
                           r"sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|"
                           r"us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))")),
        # from: https://gist.github.com/gruber/8891611
        'MASK_MENTION': re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)'),
        # from: https://stackoverflow.com/questions/2304632/regex-for-twitter-username
        'MASK_HASHTAG': re.compile(r'# ?.+?(?=\b)')  # TODO: match only hashtags with starting word boundary
    }

    @classmethod
    def clean(cls, raw_text):
        """Clean raw text. Can be overwritten by corpus specific cleaner."""
        return re.sub('\s', ' ', raw_text)

    @classmethod
    def mask(cls, unmasked_text):
        """Mask urls, @-mentions #-hashtags, emojis??? etc.

        Args:
            unmasked_text: str
        Returns: tuple containing:
            masked_text: str
            masked_strings: list of strings
        """
        masked_strings = []
        for mask, mask_pattern in cls.mask_dict.items():
            masked_strings.extend(re.findall(mask_pattern, unmasked_text))

        masked_text = unmasked_text
        for mask, mask_pattern in cls.mask_dict.items():
            masked_text = re.sub(mask_pattern, mask, masked_text)

        return masked_text, masked_strings


class SwissCrawlCleaner(Cleaner):
    pass


class SBCHCleaner(Cleaner):
    pass


class SBDECleaner(Cleaner):
    pass


class NoahCleaner(Cleaner):
    pass


class Ex3Cleaner(Cleaner):
    pass


# *****************************
# ********** Parsers **********
# *****************************


text_id = 0


class Parser:

    def __init__(self, path_in):
        self.path_in = path_in
        self.infile = open(self.path_in, 'r', encoding='utf8')

    def copy_to_main_file(self):
        """Copy the loaded file to the main file."""
        raise NotImplementedError

    @staticmethod
    def writerow(writer, cleaned_text, masked_strings, label, corpus_name):
        """Write row to output file and increase id-counter.

        Args:
            writer: csv-writer object
            cleaned_text: str
            masked_strings: list of str
            label: str
            corpus_name: str
        """
        global text_id
        writer.writerow([text_id, cleaned_text, str(masked_strings), label, corpus_name])
        if text_id % 10000:
            print('Processed line {} of {}.'.format(text_id + 1, 737628))
        text_id += 1


class Ex3Parser(Parser):

    name = 'ex3'
    out_path = 'data/main/ex3_parsed.csv'
    cleaner = Ex3Cleaner()
    num_tweets = 66921

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        infile = open(self.path_in, 'r', encoding='utf8')
        writer = csv.writer(open(self.out_path, 'w', encoding='utf8', newline='\n'))
        for line in infile:
            id_, text = json.loads(line)
            masked_text, masked_strings = self.cleaner.mask(text)
            cleaned_text = self.cleaner.clean(masked_text)
            if cleaned_text == '':
                continue
            self.writerow(writer, cleaned_text, masked_strings, '0', self.name)


class HamburgDTBParser(Parser):
    pass


class NoahParser(Parser):

    name = 'noah'
    in_dir = 'data/noah_corpus'
    out_path = 'data/main/noah_parsed.csv'
    cleaner = NoahCleaner()

    def __init__(self, path_in):
        self.path_in_dir = path_in

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        file_names = [fn for fn in os.listdir(self.path_in_dir) if fn.endswith('.xml')]
        num_files = len(file_names)
        writer = csv.writer(open(self.out_path, 'w', encoding='utf8', newline='\n'))
        for fn in file_names:
            tree = ET.parse(os.path.join(self.path_in_dir, fn))
            root = tree.getroot()
            for article in root:
                for sent in article:
                    sent_buffer = []
                    for token in sent:
                        sent_buffer.append(token.text)
                    sent_str = ' '.join(sent_buffer)
                    masked_text, masked_strings = self.cleaner.mask(sent_str)
                    cleaned_text = self.cleaner.clean(masked_text)
                    if cleaned_text == '':
                        continue
                    self.writerow(writer, cleaned_text, masked_strings, '1', self.name)


class SBCHParser(Parser):

    name = 'sb_ch'
    num_lines = 90899
    out_path = 'data/main/sb_ch_parsed.csv'
    cleaner = SBCHCleaner()

    def copy_to_main_file(self):
        """Copy the loaded file to the main file."""
        reader = csv.reader(self.infile)
        writer = csv.writer(open(self.out_path, 'w', encoding='utf8', newline='\n'))
        next(reader)
        for row in reader:
            if not row:  # continue if row/line is empty
                continue
            masked_text, masked_strings = self.cleaner.mask(row[1])
            cleaned_text = self.cleaner.clean(masked_text)
            if cleaned_text == '':
                continue
            self.writerow(writer, cleaned_text, masked_strings, '1', self.name)


class SBDEParser(Parser):

    name = 'sb_de'
    num_lines = 9983
    out_path = 'data/main/sb_de_parser.csv'
    cleaner = SBDECleaner()

    def copy_to_main_file(self):
        """Copy the loaded file to the main file."""
        reader = csv.reader(self.infile, delimiter='\t')
        writer = csv.writer(open(self.out_path, 'w', encoding='utf8', newline='\n'))
        next(reader)
        for row in reader:
            masked_text, masked_strings = self.cleaner.mask(row[3])
            cleaned_text = self.cleaner.clean(masked_text)
            if cleaned_text == '':
                continue
            self.writerow(writer, cleaned_text, masked_strings, '0', self.name)


class SwissCrawlParser(Parser):

    name = 'swisscrawl'
    num_lines = 562525
    out_path = 'data/main/swisscrawl_parsed.csv'
    cleaner = SwissCrawlCleaner()

    def copy_to_main_file(self):
        """Copy the loaded file to the main file."""
        reader = csv.reader(self.infile)
        writer = csv.writer(open(self.out_path, 'w', encoding='utf8', newline='\n'))
        next(reader)
        for row in reader:
            masked_text, masked_strings = self.cleaner.mask(row[0])
            cleaned_text = self.cleaner.clean(masked_text)
            if cleaned_text == '':
                continue
            self.writerow(writer, cleaned_text, masked_strings, '1', self.name)


def main():
    path_in = 'data/ex3_corpus/tweets.json'
    p = Ex3Parser(path_in)
    p.copy_to_main_file()

    path_in = 'data/noah_corpus/'
    p = NoahParser(path_in)
    p.copy_to_main_file()

    path_in = 'data/sb_ch_corpus/chatmania.csv'
    p = SBCHParser(path_in)
    p.copy_to_main_file()

    path_in = 'data/sb_de_corpus/downloaded.tsv'
    p = SBDEParser(path_in)
    p.copy_to_main_file()

    path_in = 'data/swisscrawl/swisscrawl-2019-11-23.csv'
    p = SwissCrawlParser(path_in)
    p.copy_to_main_file()

    try:
        os.system('rm data/main/main.csv')
    except:
        pass
    os.system('cat data/main/* > data/main/main.csv')

if __name__ == '__main__':
    main()
