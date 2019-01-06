"""
Sammy Haq
CSC 240
Final Project

Parser.py

Holds the main method. Also holds all the helper functions required to properly
run the main method. Has command line arguments:

'-v': verbose. Provides more output. Can be called again to provide even more
      runtime information ('-v -v')

'-w2vGraph': Creates and displays the w2v word map -- interactive HTML map.
"""

import json
import sys
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import FinalProjectUI as UI

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale

from keras.models import Sequential
from keras.layers import Dense

import gensim
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence

tqdm.pandas(desc="Progress: ")
# np.random.seed(123)  # for reproducibility


# Cleans a given tweet to get rid of extraneous features (such as mentions,
# hashtags, and links
def cleanDaemon(sample):
    sample = re.sub('[^a-zA-Z]', ' ', sample)
    sample = sample.lower()

    return sample


# Processing routine. Includes a progress map to let me know how far along the
# process is.
def clean(data):
    data['content'] = data['content'].progress_map(cleanDaemon)
    return data


def extract(data):
    samples = []
    labels = []

    for i in tqdm(range(0, len(data))):
        samples.append(data.content[i])
        labels.append(int(data.annotation[i]['label'][0]))

    return samples, labels


def labelize(data, labelMarker):
    labelized = []
    for i, j in tqdm(enumerate(data)):
        label = '%s_%s' % (labelMarker, i)
        labelized.append(LabeledSentence(j.split(" "), [label]))
    return labelized


def rebuildTweet(tweet, w2v, tfidf, size):
    tweetVector = np.zeros(size).reshape((1, size))
    count = 0.

    for word in tweet:
        try:
            tweetVector += w2v[word].reshape((1, size)) * tfidf[word]
            count += 1
        except KeyError:
            continue

    if count != 0:
        tweetVector /= count

    return tweetVector


###############
# MAIN METHOD #
###############
def main():
    verbose = False
    verboseVerbose = False

    if ("-v" in sys.argv):
        if (sys.argv.count("-v") == 1):
            verbose = True
        else:
            verboseVerbose = True
            verbose = True

    # Setting to show intermediary steps and progress.
    ui = UI.FinalProjectUI(True, True)
    ui.titleLabel()

    # Grabbing data and putting into folder.
    data = pd.read_json("data/Dataset for Detection of Cyber-Trolls.json",
                        lines=True)

    # Processing data. Trimming and removing everything but letters.
    # Also makes everything lowercase.
    ui.cleaningLabel(len(data))
    data = clean(data)
    ui.done()

    # Printing data snapshot.
    ui.printTable(data)

    # Extracting samples and labels from data.
    ui.extractingLabel()
    samples, labels = extract(data)
    ui.done()
    ui.printSamplesLabelsTable(samples, labels)

    x_train, x_test, y_train, y_test = train_test_split(samples,
                                                        labels, test_size=0.2)

    # Preparing to feed into words2vec..
    ui.labelizeInit()
    x_train = labelize(x_train, 'TRAIN')
    x_test = labelize(x_test, 'TEST')
    ui.done()

    # Training words2vec model, each vector defined as a size of 200
    ui.words2VecInit()
    w2v = Word2Vec(size=200, min_count=10)
    w2v.build_vocab([x.words for x in tqdm(x_train)])
    w2v.train([x.words for x in tqdm(x_train)],
              total_examples=w2v.corpus_count,
              epochs=w2v.iter)
    ui.done()

    if ("-w2vGraph" in sys.argv):
        ui.words2VecGraphInit()
        ui.graphWords2VecMap(w2v, w2v.wv.vocab.keys()[:5000])
        ui.done()

    # TF-IDF Statistic used for word weighting in sentences
    ui.tfidfInit()
    tfidfVectorizer = TfidfVectorizer(analyzer=lambda x: x)
    matrix = tfidfVectorizer.fit_transform([x.words for x in tqdm(x_train)])
    tfidf = dict(zip(tfidfVectorizer.get_feature_names(),
                     tfidfVectorizer.idf_))
    ui.done()

    # Applying TF-IDF to words to create word vectors.
    ui.tfidfApply()
    train_tweetData = np.concatenate(
                      [rebuildTweet(item, w2v, tfidf, 200) for item in tqdm(
                       map(lambda x: x.words, x_train))])

    test_tweetData = np.concatenate(
                      [rebuildTweet(item, w2v, tfidf, 200) for item in tqdm(
                       map(lambda x: x.words, x_test))])
    ui.done()

    # Scaling both train and test vectors to have a zero mean and standard dev
    ui.scaleInit()
    train_tweetData = scale(train_tweetData)
    test_tweetData = scale(test_tweetData)
    ui.done()

    ui.neuralNetworkInit()
    model = Sequential()
    model.add(Dense(32, activation='softmax', input_dim=200))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    ui.done()

    ui.neuralNetworkTrain()
    model.fit(train_tweetData,
              np.array(y_train),
              epochs=100, batch_size=32, verbose=2)
    ui.done()

    ui.neuralNetworkResults()
    testResults = model.predict(np.array(test_tweetData))

    # Test results are nonbinary, so they must be rounded.
    print(confusion_matrix(y_test, testResults.round()))
    print(classification_report(y_test, testResults.round()))
    ui.done()


main()
