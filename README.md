# shared_task
Swiss German Language Detection

Collaborators: Janis Goldzycher, Jonathan Schaber

## Timeline:

- 22.03.2020: Bigram Model trained/finished
- 29.03.2020: Embedding Based Model traind/finished
- 20.03.2020: Test Set Release
- 27.03.2020: Experimental Results Due
- 03.04.2020: Publication of Evaluation Results
- 14.04.2020: System Description Submission
- 21.04.2020: Acceptance Notification
- 05.05.2020: Camera Ready
- 23-24.06.2020: Swiss Text Conference

TODOs for week 17.-22. Feb:
- [x] write script to split training/validation data
- [ ] write script for evaluation
- [x] write bigram based model
- [x] train and test bigram based model

TODOs for week 23.-29. Feb:
- [ ] train and test embedding based model

Format of system output:
- csv file
- columns: id, label
- optional: confidence as additional column

## Execution Instructions

1. activate the conda environment
2. `python corpus_parser.py`
3. `python split_corpus.py -r 0.8`
4. `python3 generate_bigram_repr.py -g -m train_bigram_to_dim_mapping.json -i data/main/train_main.csv -o data/main/train_main_bigr_repr.csv`
5. `python3 generate_bigram_repr.py -m train_bigram_to_dim_mapping.json -i data/main/dev_main.csv -o data/main/dev_main_bigr_repr.csv`
6. Some call for training
7. Some call for evaluation

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
  1. filter out text that is mostly not latin1 (ignore emojis) -> negative for sure
  2. clean/mask text remaining text examples
  3. pass remaining text examples to trained model (and only train model on such languages)

#### Ideas for models

  - bigram based
    - "multi"-hot-encoding for bigrams
    - problems: sparsity, high dimenionality
    - for classification on top: svm, mlp, CNNs (hierarchical)
  - character embedding based
    - compute character embeddings using char-lang-model or cbow/skip-gram for on char-level?
    - for classification on top: CNN (hierarchical, mutiple channels, dilation etc)  
    
    
## Questions

- What ratio of swiss german vs not-swiss german should we target? (depends on distribution of test data?)
- Invest more time in collecting data or tweaking models (do we need more swiss german data)?
- other model ideas?
- other representation ideas?
- what are the SOTA performances in language detection?
- Do you need data who contributed what? What with pair programming? etc
- Can the submitted paper count as the report for the pp?


## Additional Corpora
- Hamburd Tree Bank: already downloaded
- Leipzig corpora (many languages to choose from): https://wortschatz.uni-leipzig.de/en/download


## Relevant Literature and Resources

### Languate Detection
https://arxiv.org/pdf/1804.08186.pdf
https://dbs.cs.uni-duesseldorf.de/lehre/bmarbeit/barbeiten/ba_panich.pdf
https://www.slideshare.net/shuyo/language-detection-library-for-java

### Character Embeddings
https://www.aclweb.org/anthology/S19-1008/
https://www.depends-on-the-definition.com/lstm-with-char-embeddings-for-ner/
https://hackernoon.com/chars2vec-character-based-language-model-for-handling-real-world-texts-with-spelling-errors-and-a3e4053a147d
https://github.com/euler16/CharRNN
https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch

### Swiss Dialect Identification
https://github.com/bricksdont/swiss-dialect-identification

