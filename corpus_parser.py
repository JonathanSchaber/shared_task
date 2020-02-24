import os
import re
import csv
import json
import xml.etree.ElementTree as ET
from abc import ABC

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
        'MASK_MENTION': re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-_.]))@([A-Za-z]+[A-Za-z0-9-_]+)'),
        # from: https://stackoverflow.com/questions/2304632/regex-for-twitter-username
        'MASK_HASHTAG': re.compile(r'# ?.+?(?=\b)')  # TODO: match only hashtags with starting word boundary
    }

    @classmethod
    def clean(cls, raw_text):
        """Clean raw text. Can be overwritten by corpus specific cleaner."""
        return re.sub(r'\s', ' ', raw_text)

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


class HamburgTBCleaner(Cleaner):
    pass


class LeipzigCleaner(Cleaner):
    pass


class LeipzigCleanerBAR(LeipzigCleaner):
    pass


class LeipzigCleanerDE(LeipzigCleaner):
    pass


class LeipzigCleanerEN(LeipzigCleaner):
    pass


class LeipzigCleanerFR(LeipzigCleaner):
    pass


class LeipzigCleanerFRR(LeipzigCleaner):
    pass


class LeipzigCleanerITA(LeipzigCleaner):
    pass


class LeipzigCleanerLMO(LeipzigCleaner):
    pass


class LeipzigCleanerLTZ(LeipzigCleaner):
    pass


class LeipzigCleanerNDS(LeipzigCleaner):
    pass


class LeipzigCleanerNLD(LeipzigCleaner):
    pass


class LeipzigCleanerNOR(LeipzigCleaner):
    pass


class LeipzigCleanerSWE(LeipzigCleaner):
    pass


class LeipzigCleanerYID(LeipzigCleaner):
    pass


class LeipzigCleanerGSW(LeipzigCleaner):
    pass


# *****************************
# ********** Parsers **********
# *****************************


text_id = 0
lang_to_label = json.load(open('lang_to_label_mappings.json', 'r', encoding='utf8'))


class Parser:
    num_lines_overall = 737628

    def copy_to_main_file(self):
        """Copy the loaded file to the main file."""
        raise NotImplementedError

    @classmethod
    def writerow(cls, writer, cleaned_text, masked_strings, label_binary, label_ternary, label_finegrained,
                 corpus_name):
        """Write row to output file and increase id-counter.

        Args:
            writer: csv-writer object
            cleaned_text: str
            masked_strings: list of str
            label_binary: str
            label_ternary: str
            label_finegrained: str
            corpus_name: str
        """
        global text_id
        writer.writerow(
            [text_id, cleaned_text, str(masked_strings), label_binary, label_ternary, label_finegrained, corpus_name])
        if text_id % 10000:
            print('Processed line {} of {}.'.format(text_id + 1, cls.num_lines_overall))
        text_id += 1


class LeipzigParser(Parser):

    label_binary = 'default'
    label_ternary = 'default'
    label_finegrained = 'default'

    @classmethod
    def _copy_to_main_file(cls, path_in, path_out, cleaner, name):
        csv_writer = csv.writer(open(path_out, 'w', encoding='utf8'))
        with open(path_in, 'r', encoding='utf8') as f:
            for line in f:
                text_id, text = line.strip('\n').split('\t')
                masked_text, masked_strings = cleaner.mask(text)
                cleaned_text = cleaner.clean(masked_text)
                if cleaned_text == '':
                    continue
                cls.writerow(csv_writer, cleaned_text, masked_strings, cls.label_binary, cls.label_ternary,
                             cls.label_finegrained, name)


class LeipzigParserBAR(LeipzigParser):
    """For Bavarian."""

    path_in = 'data/leipzig_bar/bar_wikipedia_2010_30K-sentences.txt'
    path_out = 'data/main/leipzig_bar_parsed.csv'
    language = 'bavarian'
    corpus_name = 'leipzig_bar'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['german']
    label_finegrained = lang_to_label['finegrained']['bavarian']
    cleaner = LeipzigCleanerBAR()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserDE(LeipzigParser):
    """For German."""

    path_in = 'data/leipzig_de/deu_mixed-typical_2011_300K-sentences.txt'
    path_out = 'data/main/leipzig_de_parsed.csv'
    language = 'german'
    corpus_name = 'leipzig_de'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary'][language]
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerDE()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserEN(LeipzigParser):
    """For English."""

    path_in = 'data/leipzig_en/eng_news_2016_300K-sentences.txt'
    path_out = 'data/main/leipzig_en_parsed.csv'
    language = 'english'
    corpus_name = 'leipzig_en'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['other']
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerEN()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserFR(LeipzigParser):
    """For French."""

    path_in = 'data/leipzig_fr/fra_mixed_2009_300K-sentences.txt'
    path_out = 'data/main/leipzig_fr_parsed.csv'
    language = 'french'
    corpus_name = 'leipzig_fr'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['other']
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerFR()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserFRR(LeipzigParser):
    """For North Frisian."""

    path_in = 'data/leipzig_frr/frr_wikipedia_2016_10K-sentences.txt'
    path_out = 'data/main/leipzig_frr_parsed.csv'
    language = 'northern_frisian'
    corpus_name = 'leipzig_frr'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['german']  # TODO: Should it count as german?
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerFRR()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserITA(LeipzigParser):
    """For Italian."""

    path_in = 'data/leipzig_ita/ita_mixed-typical_2017_300K-sentences.txt'
    path_out = 'data/main/leipzig_ita_parsed.csv'
    language = 'italian'
    corpus_name = 'leipzig_ita'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['other']
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerITA()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserLMO(LeipzigParser):
    """For Lombardic."""

    path_in = 'data/leipzig_lmo/lmo_wikipedia_2016_30K-sentences.txt'
    path_out = 'data/main/leipzig_lmo_parsed.csv'
    language = 'lombard'
    corpus_name = 'leipzig_lmo'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['other']
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerLMO()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserLTZ(LeipzigParser):
    """For Luxembourgish."""

    path_in = 'data/leipzig_ltz/ltz_newscrawl_2016_300K-sentences.txt'
    path_out = 'data/main/leipzig_ltz_parsed.csv'
    language = 'luxembourgish'
    corpus_name = 'leipzig_ltz'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['other']
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerLTZ()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserNDS(LeipzigParser):
    """For low german (niedersächsisch)."""

    path_in = 'data/leipzig_nds/nds_wikipedia_2016_100K-sentences.txt'
    path_out = 'data/main/leipzig_nds_parsed.csv'
    language = 'low_german'
    corpus_name = 'leipzig_nds'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['german']  # Should it count as german???
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerNDS()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserNLD(LeipzigParser):
    """For Dutch."""

    path_in = 'data/leipzig_nld/nld_mixed_2012_300K-sentences.txt'
    path_out = 'data/main/leipzig_nld_parsed.csv'
    language = 'dutch'
    corpus_name = 'leipzig_nld'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['other']
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerNLD()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserNOR(LeipzigParser):
    """For Norwegian."""

    path_in = 'data/leipzig_nor/nor_wikipedia_2016_300K-sentences.txt'
    path_out = 'data/main/leipzig_nor_parsed.csv'
    language = 'norwegian'
    corpus_name = 'leipzig_nor'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['other']
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerNOR()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserSWE(LeipzigParser):
    """For Swedish.."""

    path_in = 'data/leipzig_swe/swe_wikipedia_2016_300K-sentences.txt'
    path_out = 'data/main/leipzig_swe_parsed.csv'
    language = 'swedish'
    corpus_name = 'leipzig_swe'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['other']
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerSWE()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserYID(LeipzigParser):
    """For Yiddish.."""

    path_in = 'data/leipzig_yid/yid_wikipedia_2016_30K-sentences.txt'
    path_out = 'data/main/leipzig_yid_parsed.csv'
    language = 'yiddish'
    corpus_name = 'leipzig_yid'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['other']
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerYID()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class LeipzigParserGSW(LeipzigParser):
    """For Swiss German (Leipzig News Corpus).."""

    path_in = 'data/leipzig_gsw/yid_wikipedia_2016_30K-sentences.txt'
    path_out = 'data/main/leipzig_gsw_parsed.csv'
    language = 'swiss_german'
    corpus_name = 'leipzig_gsw'
    label_binary = lang_to_label['binary'][language]
    label_ternary = lang_to_label['ternary'][language]
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = LeipzigCleanerGSW()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        self._copy_to_main_file(self.path_in, self.path_out, self.cleaner, self.corpus_name)


class Ex3Parser(Parser):
    path_in = 'data/ex3_corpus/tweets.json'
    path_out = 'data/main/ex3_parsed.csv'
    language = 'various'
    corpus_name = 'ex3'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['other']
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = Ex3Cleaner()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        infile = open(self.path_in, 'r', encoding='utf8')
        writer = csv.writer(open(self.path_out, 'w', encoding='utf8', newline='\n'))
        for line in infile:
            id_, text = json.loads(line)
            masked_text, masked_strings = self.cleaner.mask(text)
            cleaned_text = self.cleaner.clean(masked_text)
            if cleaned_text == '':
                continue
            self.writerow(writer, cleaned_text, masked_strings, self.label_binary,
                          self.label_ternary, self.label_finegrained, self.corpus_name)


class HamburgDTBParser(Parser):
    path_in = 'data/hamburg_dep_treebank/hamburg-dependency-treebank-conll/'
    filenames = ['part_A.conll', 'part_B.conll', 'part_C.conll']
    path_out = 'data/main/hamburgtb_parsed.csv'
    language = 'german'
    corpus_name = 'hamburgtb'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary']['german']
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = HamburgTBCleaner()

    def clean_and_write(self, csv_writer, words):
        sent_str = ' '.join(words)
        masked_text, masked_strings = self.cleaner.mask(sent_str)
        cleaned_text = self.cleaner.clean(masked_text)
        if cleaned_text == '':
            return
        self.writerow(csv_writer, cleaned_text, masked_strings, self.label_binary,
                      self.label_ternary, self.label_finegrained, self.corpus_name)

    def copy_to_main_file(self):
        """Copy parsed contents of all conll-files to the main file (csv) one sentence per row."""
        fpaths_in = [os.path.join(self.path_in, fn) for fn in self.filenames]
        fout = open(self.path_out, 'w', encoding='utf8')
        csv_writer = csv.writer(fout)
        for fp in fpaths_in:
            with open(fp, 'r', encoding='utf8') as f:
                words = []
                for line in f:
                    if line == '\n':
                        self.clean_and_write(csv_writer, words)
                        words = []
                    else:
                        word = line.split('\t')[1]
                        words.append(word)
                self.clean_and_write(csv_writer, words)


class NoahParser(Parser):
    path_in = 'data/noah_corpus'
    out_path = 'data/main/noah_parsed.csv'
    language = 'swiss_german'
    corpus_name = 'noah'
    label_binary = lang_to_label['binary'][language]
    label_ternary = lang_to_label['ternary'][language]
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = NoahCleaner()

    def copy_to_main_file(self):
        """Copy parsed contents of all xml-files to the main file (csv) one sentence per row."""
        file_names = [fn for fn in os.listdir(self.path_in) if fn.endswith('.xml')]
        # num_files = len(file_names)
        writer = csv.writer(open(self.out_path, 'w', encoding='utf8', newline='\n'))
        for fn in file_names:
            tree = ET.parse(os.path.join(self.path_in, fn))
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
                    self.writerow(writer, cleaned_text, masked_strings, self.label_binary,
                                  self.label_ternary, self.label_finegrained, self.corpus_name)


class SBCHParser(Parser):
    num_lines = 90899
    path_in = 'data/sb_ch_corpus/chatmania.csv'
    out_path = 'data/main/sb_ch_parsed.csv'
    language = 'swiss_german'
    corpus_name = 'sb_ch'
    label_binary = lang_to_label['binary'][language]
    label_ternary = lang_to_label['ternary'][language]
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = SBCHCleaner()

    def copy_to_main_file(self):
        """Copy the loaded file to the main file."""
        reader = csv.reader(open(self.path_in, 'r', encoding='utf8'))
        writer = csv.writer(open(self.out_path, 'w', encoding='utf8', newline='\n'))
        next(reader)
        for row in reader:
            if not row:  # continue if row/line is empty
                continue
            masked_text, masked_strings = self.cleaner.mask(row[1])
            cleaned_text = self.cleaner.clean(masked_text)
            if cleaned_text == '':
                continue
            self.writerow(writer, cleaned_text, masked_strings, self.label_binary,
                          self.label_ternary, self.label_finegrained, self.corpus_name)


class SBDEParser(Parser):
    num_lines = 9983
    path_in = 'data/sb_de_corpus/downloaded.tsv'
    out_path = 'data/main/sb_de_parsed.csv'
    language = 'german'
    corpus_name = 'sb_de'
    label_binary = lang_to_label['binary']['other']
    label_ternary = lang_to_label['ternary'][language]
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = SBDECleaner()

    def copy_to_main_file(self):
        """Copy the loaded file to the main file."""
        reader = csv.reader(open(self.path_in, 'r', encoding='utf8'), delimiter='\t')
        writer = csv.writer(open(self.out_path, 'w', encoding='utf8', newline='\n'))
        next(reader)
        for row in reader:
            masked_text, masked_strings = self.cleaner.mask(row[3])
            cleaned_text = self.cleaner.clean(masked_text)
            if cleaned_text == '':
                continue
            self.writerow(writer, cleaned_text, masked_strings, self.label_binary,
                          self.label_ternary, self.label_finegrained, self.corpus_name)


class SwissCrawlParser(Parser):
    num_lines = 562525
    path_in = 'data/swisscrawl/swisscrawl-2019-11-23.csv'
    path_out = 'data/main/swisscrawl_parsed.csv'
    language = 'swiss_german'
    corpus_name = 'swisscrawl'
    label_binary = lang_to_label['binary'][language]
    label_ternary = lang_to_label['ternary'][language]
    label_finegrained = lang_to_label['finegrained'][language]
    cleaner = SwissCrawlCleaner()

    def copy_to_main_file(self):
        """Copy the loaded file to the main file."""
        reader = csv.reader(open(self.path_in, 'r', encoding='utf8'))
        writer = csv.writer(open(self.path_out, 'w', encoding='utf8', newline='\n'))
        next(reader)
        for row in reader:
            masked_text, masked_strings = self.cleaner.mask(row[0])
            cleaned_text = self.cleaner.clean(masked_text)
            if cleaned_text == '':
                continue
            self.writerow(writer, cleaned_text, masked_strings, self.label_binary,
                          self.label_ternary, self.label_finegrained, self.corpus_name)


def main():
    parsers = [Ex3Parser, NoahParser, SBCHParser, SBDEParser, SwissCrawlParser,
               HamburgDTBParser, LeipzigParserBAR, LeipzigParserDE, LeipzigParserEN,
               LeipzigParserFR, LeipzigParserFRR, LeipzigParserITA, LeipzigParserLMO,
               LeipzigParserLTZ, LeipzigParserNDS, LeipzigParserNLD, LeipzigParserNOR,
               LeipzigParserSWE, LeipzigParserYID]

    for parser_type in parsers:
        parser = parser_type()
        parser.copy_to_main_file()

    try:
        os.system('rm data/main/main.csv')
    except:
        pass
    os.system('cat data/main/*_parsed.csv > data/main/main.csv')


if __name__ == '__main__':
    main()
