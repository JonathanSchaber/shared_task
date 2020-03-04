# Swiss German Language Detection

Swiss German Language Detection

Collaborators: Janis Goldzycher, Jonathan Schaber

## Timeline:

- 22.03.2020: Bigram Model trained/finished
- 29.03.2020: Embedding Based Model trained/finished
- 20.03.2020: Test Set Release
- 27.03.2020: Experimental Results Due
- 03.04.2020: Publication of Evaluation Results
- 14.04.2020: System Description Submission
- 21.04.2020: Acceptance Notification
- 05.05.2020: Camera Ready
- 23-24.06.2020: Swiss Text Conference

TODOs for week 17.-22. Feb:
- [x] write script to split training/validation data
- [x] write script for evaluation
- [x] write bigram based model
- [x] train and test bigram based model
- [x] get balanced training data -> Janis
- [ ] train svm over subset of balanced training data to get baseline -> Janis
- [ ] implement filter mechanism; all on swiss keyboard; test on swiss data, if stuff is filtered out -> Jonathan
- [ ] test if filter mechanism produces errors -> Jonathan
- [ ] train character embeddings
- [ ] train CNN

TODOs for week 23.-29. Feb:
- [ ] train and test embedding based model

Format of system output:
- csv file
- columns: id, text, masked-items, label-binary, label-ternary label-finegrained, corpus-name
- swiss german always has the label 0
- optional: confidence as additional column

Format of bigram repr files:
text_id KOMMA label_binary KOMMA label_ternary KOMMA label_finegrained KOMMA Representations

## Corpus Statistics

### Overall
- Number of examples (line of main.csv): 3'798'459
- Number positive examples (ch): 760'729 (`grep -Pc '\d+,.*,.*,0,.*$' main.csv`)
- Number of negative examples (other): 2'937'735 (`grep -Pc '\d+,.*,.*,0,.*$' main.csv`) <- needs update
- Number of standard german examples (part of other): hamburgtb + sbde + leipzig_de + leipzig_bar + a little ex3 = ca. 600'000

### Subcorpora

Number of examples:
- ex3-corpus: 66'921
- hamburgtb: 261'821
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
- leipzig_ita: 300'000
- leipzig_lmo: 30'000
- leipzig_itz: 300'000
- leipzig_nds: 100'000
- leipzig_nld: 300'000
- leipzig_nor: 300'000
- leipzig_swe: 300'000
- leipzig_yid: 30'000


## Execution Instructions

### Local 
1. activate the conda environment
2. `python corpus_parser.py`
3. `python split_corpus.py`
4. `python3 generate_bigram_repr.py -g -m train_bigram_to_dim_mapping.json -i data/main/train_main.csv -o data/main/train_main_bigr_repr.csv`
5. `python3 generate_bigram_repr.py -m train_bigram_to_dim_mapping.json -i data/main/dev_main.csv -o data/main/dev_main_bigr_repr.csv`
6. `python3 create_train_subcorpus.py -i data/main/train_main_bigr_repr.csv -g <granularity> -n num_ex_per_clas`
    - outputfile: `data/main/train_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv`
7. `python3 create_train_subcorpus.py -i data/main/dev_main_bigr_repr.csv -g <granularity> -n num_ex_per_clas`
    - outputfile: `data/main/dev_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv`
8. `python3 bigram_based_models.py -t data/main/train_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv -d data/main/dev_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv -o results/`

### Server
1. activate the conda environment
2. `python corpus_parser.py -s`
3. `python split_corpus.py -s -i /home/user/jgoldz/storage/shared_task/data/main/main.csv -o /home/user/jgoldz/storage/shared_task/data/main/`
4. `python3 generate_bigram_repr.py -g -m train_bigram_to_dim_mapping.json -i /home/user/jgoldz/storage/shared_task/data/main/train_main.csv -o /home/user/jgoldz/storage/shared_task/data/main/train_main_bigr_repr.csv`
5. `python3 generate_bigram_repr.py -m train_bigram_to_dim_mapping.json -i /home/user/jgoldz/storage/shared_task/data/main/dev_main.csv -o /home/user/jgoldz/storage/shared_task/data/main/dev_main_bigr_repr.csv`
6. `python3 create_train_subcorpus.py -i /home/user/jgoldz/storage/shared_task/data/main/train_main_bigr_repr.csv -g <granularity> -n <num_ex_per_clas>`
    - outputfile: `/home/user/jgoldz/storage/shared_task/data/main/train_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv`
7. `python3 create_train_subcorpus.py -i /home/user/jgoldz/storage/shared_task/data/main/dev_main_bigr_repr.csv -g <granularity> -n <num_ex_per_clas>`
    - outputfile: `/home/user/jgoldz/storage/shared_task/data/main/dev_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv`
8. `python3 bigram_based_models.py -t /home/user/jgoldz/storage/shared_task/data/main/train_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv -d /home/user/jgoldz/storage/shared_task/data/main/dev_main_bigr_repr_<granularity>_<num_ex_per_clas>.csv -o results/`

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

#### Ideas for models

  - bigram based
    - "multi"-hot-encoding for bigrams
    - problems: sparsity, high dimensionality
    - for classification on top: svm, mlp, CNNs (hierarchical)
  - character embedding based
    - compute character embeddings using char-lang-model or cbow/skip-gram for on char-level?
    - for classification on top: CNN (hierarchical, multiple channels, dilation etc)  
    
    
## Questions

### Organizational
- Can the submitted paper count as the report for the pp? yes

### Technical
- what are the SOTA performances in language detection? -> over 99%
- What ratio of swiss german vs not-swiss german should we target? (depends on distribution of test data?)
- Invest more time in collecting data or tweaking models (do we need more swiss german data)?
- Encode with padding or rnn-hidden representations?
- other model ideas?
- other representation ideas?

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