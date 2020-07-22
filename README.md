# Swiss German Language Detection

Swiss German Language Detection

Collaborators: Janis Goldzycher, Jonathan Schaber

Paper Link: http://ceur-ws.org/Vol-2624/germeval-task2-paper3.pdf

## Project

This repo contains the code for our system submission for the Swiss German language detection shared task, part of the GermEval 2020 Campaign, held at the SwissText & KONVENS 2020 conference.
In the file `neural_models.py` you find an overview over different architectures we experimented with.
The submitted model is the one defined in the `SeqToLabelModelOnlyHiddenBiDeepOriginal` class.

Note that to reproduce our results you need to download the corpora and reproduce the corresponding directory structure as well.
For detailed information, please see the [Execution Instructions](#execution-instructions)

## Corpus Statistics

### Overall
- Number of examples (line of main.csv): 4'071'017
- Number positive examples (ch): 780'507 (`grep -Pc '\d+,.*,.*,0,.*$' main.csv`)
- Number of negative examples (other): 3'290'515 (`grep -Pc '\d+,.*,.*,1,.*$' main.csv`)
- Number of standard german examples (part of other): hamburgtb + sbde + leipzig_de + leipzig_bar + a little ex3 = ca. 600'000

### Subcorpora

Number of examples:
- ex3-corpus: 66'921
- hamburgtb: 261'821
- hrWa: 10'000
- sbde: 9'982
- noah: 7'303
- sbch: 90'897
- swisscrawl: 562'524
- leipzig_bar: 30'000 
- leipzig_de: 299'994
- leipzig_en: 300'000
- leipzig_fr: 299'017
- leipzig_frr: 10'000
- leipzig_fry: 100'000
- leipzig_gsw: 100'000
- leipzig_hbs: 100'000
- leipzig_ita: 300'000
- leipzig_ita: 300'000
- leipzig_lmo: 30'000
- leipzig_itz: 300'000
- leipzig_nds: 100'000
- leipzig_nld: 300'000
- leipzig_nor: 300'000
- lepizig_por: 100'000
- leipzig_ron: 100'000
- leipzig_swe: 300'000
- leipzig_spa: 100'000
- leipzig_srp: 100'000
- leipzig_tgl: 100'000
- leipzig_yid: 30'000


## Execution Instructions

### Local 
1. activate the conda environment
2. `python corpus_parser.py`
3. `python split_corpus.py`

#### For training a neural model
4. `python neural_models.py -c <path_to_config> -d <device> -g <gpu-core> -l <location>`
5. `python predict.py -m <model> -t <type> -i <input file> -o <output file> -c <config file> -g <gpu-core>`
6. `python evaluation.py -p <predicted-file>`

#### For training a character bigram-based SVM
7. `python generate_bigram_repr.py -g -m train_bigram_to_dim_mapping.json -i data/main/train_main.csv -o data/main/train_main_bigr_repr.csv`
8. `python generate_bigram_repr.py -m train_bigram_to_dim_mapping.json -i data/main/dev_main.csv -o data/main/dev_main_bigr_repr.csv`
9. `python create_train_subcorpus.py -i data/main/train_main_bigr_repr.csv -g <granularity> -n num_ex_per_clas`
    - outputfile: `data/main/train_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv`
10. `python create_train_subcorpus.py -i data/main/dev_main_bigr_repr.csv -g <granularity> -n num_ex_per_clas`
    - outputfile: `data/main/dev_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv`
11. `python bigram_based_models.py -t data/main/train_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv -d data/main/dev_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv -o results/`

### Server
1. activate the conda environment
2. `python corpus_parser.py -s`
3. `python split_corpus.py -s -i /home/user/jgoldz/storage/shared_task/data/main/main.csv -o /home/user/jgoldz/storage/shared_task/data/main/`


#### For training a neural model
4. `python neural_models.py -c <path_to_config> -d <device> -g <gpu-core> -l <location>`
5. `python predict.py -m <model> -t <type> -i <input file> -o <output file> -c <config file> -g <gpu-core>`
6. `python evaluation.py -p <predicted-file>`


#### For training a character bigram-based SVM
7. `python generate_bigram_repr.py -g -m train_bigram_to_dim_mapping.json -i /home/user/jgoldz/storage/shared_task/data/main/train_main.csv -o /home/user/jgoldz/storage/shared_task/data/main/train_main_bigr_repr.csv`
8. `python generate_bigram_repr.py -m train_bigram_to_dim_mapping.json -i /home/user/jgoldz/storage/shared_task/data/main/dev_main.csv -o /home/user/jgoldz/storage/shared_task/data/main/dev_main_bigr_repr.csv`
9. `python create_train_subcorpus.py -i /home/user/jgoldz/storage/shared_task/data/main/train_main_bigr_repr.csv -g <granularity> -n <num_ex_per_clas>`
    - outputfile: `/home/user/jgoldz/storage/shared_task/data/main/train_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv`
10. `python create_train_subcorpus.py -i /home/user/jgoldz/storage/shared_task/data/main/dev_main_bigr_repr.csv -g <granularity> -n <num_ex_per_clas>`
    - outputfile: `/home/user/jgoldz/storage/shared_task/data/main/dev_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv`
11. `python bigram_based_models.py -t /home/user/jgoldz/storage/shared_task/data/main/train_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv -d /home/user/jgoldz/storage/shared_task/data/main/dev_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv -o results/`

## Linguistic Considerations

- linguistic intuition: most difficult languages to distinguish from swiss german are:
  - Standard German
  - Luxemburgish
  - Dutch
  - French
  - English
  - -> these languages should have strong representation in the training data
- assumption: swiss german text will mostly use latin1 characters
- -> text that mostly consists of other characters (after cleaning of urls etc.) can be ruled out in a 
rule based way (without classification)
- Thus, the structure:
  1. filter out text that is mostly not lGood video from oral presentation: https://vimeo.com/384767351atin1 (ignore emojis) -> negative for sure
  2. clean/mask text remaining text examples
  3. pass remaining text examples to trained model (and only train model on such languages)

## Additional Corpora
- Hamburg Tree Bank: already downloaded
- Leipzig corpora (many languages to choose from): https://wortschatz.uni-leipzig.de/en/download
- schwäbisch: https://www.schwaebisch-schwaetza.de/datenschutzerklaerung.htm


## Relevant Literature and Resources

### Language Detection
- kocmi: https://arxiv.org/pdf/1701.03338.pdf
- survey: https://arxiv.org/pdf/1804.08186.pdf
- http://ceur-ws.org/Vol-1228/
- http://ceur-ws.org/Vol-1228/tweetlid-1-gamallo.pdf -> winner of above, best model: naive bayes with bigrams and trigrams for prefix and suffix
- https://dbs.cs.uni-duesseldorf.de/lehre/bmarbeit/barbeiten/ba_panich.pdf
- https://www.slideshare.net/shuyo/language-detection-library-for-java

### Character Embeddings
- https://www.aclweb.org/anthology/S19-1008/
- https://www.depends-on-the-definition.com/lstm-with-char-embeddings-for-ner/
- https://hackernoon.com/chars2vec-character-based-language-model-for-handling-real-world-texts-with-spelling-errors-and-a3e4053a147d
- https://github.com/euler16/CharRNN
- https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch

### Swiss Dialect Identification
- https://github.com/bricksdont/swiss-dialect-identification

### Libraries
- for sequence classification: https://github.com/pytorch/fairseq

## Instructions for Adding a Corpus

1. Create a directory in `data/` and copy files into directory.
2. Add language-label mappings in `lang_to_label_mappings.json` if necessary.
3. Create a CorpusCleaner. Can just be a empty class.
4. Create a Corpus Parser. Needs to implement `copy_to_main_file()` and at least have the following attributes: 
    * `path_in`
    * `path_out` 
    * `language`
    * `corpus_name`
    * `label_binary`
    * `label_ternary`
    * `label_finegrained`
    * `cleaner`
5. Include parser in `parsers`-list in `main`.
6. Update corpus statistics in this readme.
