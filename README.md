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
- [ ] write bigram based model
- [ ] train and test bigram based model

TODOs for week 23.-29. Feb:
- [ ] train and test embedding based model

Format of system output:
- csv file
- columns: id, label
- optional: confidence as additional column

##  Execution Instructions

1. activate the conda environment
2. `python corpus_parser.py`
3. `python split_corpus.py -r 0.8`
4. `python3 generate_bigram_repr.py -g -m train_bigram_to_dim_mapping.json -i data/main/train_main.csv -o data/main/train_main_bigr_repr.csv`
5. `python3 generate_bigram_repr.py -i -m train_bigram_to_dim_mapping.json data/main/dev_main.csv -o data/main/dev_main_bigr_repr.csv`
6. Some call for training
7. Some call for evaluation