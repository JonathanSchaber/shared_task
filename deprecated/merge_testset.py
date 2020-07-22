import csv

reader_tweets = csv.reader(open('data/main/test_tweets.full.with_missing.csv'))
reader_labels = csv.reader(open('../gswid2020/data/test_tweets.labelled.csv'))
writer_gold = csv.writer(open('data/main/test_tweets.full.with_missing.gold_labels.csv', 'w', encoding='utf8'))
id_to_label = {}
id_to_tweet = {}
id_tweet_label = {}  # {id: (tweet, label)}
next(reader_tweets)
next(reader_labels)
for row in reader_tweets:
    twid, tweet = row
    id_to_tweet[twid] = tweet

for row in reader_labels:
    twid, label = row
    id_to_label[twid] = 0 if label == 'gsw' else 1

for twid in id_to_tweet:
    writer_gold.writerow([twid, id_to_tweet[twid], [], id_to_label[twid], id_to_label[twid], id_to_label[twid], 'testset'])