"""
Sammy Haq
CSC 240
Final Project

FinalProjectUI.py

Provides all the UI components that print to terminal. All print methods are
here to ensure Parser.py stays uncluttered. ALso uses tqdm in order to provide
progress bars (useful for runtime and ensuring nothing crashed).
"""

import pandas as pd
from tqdm import tqdm
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from sklearn.manifold import TSNE


class FinalProjectUI:
    def __init__(self, verbose, verboseVerbose):
        self.verbose = verbose
        self.verboseVerbose = verboseVerbose

    def isVerbose(self):
        return self.verbose

    def isVerboseVerbose(self):
        return self.verboseVerbose

    def titleLabel(self):
        print("\n**")
        print("* Sammy Haq\n* CSC240\n* Final Project\n* Due 12/12/2018")
        print("**")

    def cleaningLabel(self, size):
        if self.verbose:
            toPrint = "* Cleaning dataset of size " + str(size) + ".."

            asteriskBuffer = ""
            for i in range(0, len(toPrint)):
                asteriskBuffer = asteriskBuffer + "*"

            print("\n" + asteriskBuffer)
            print(toPrint)
            print(asteriskBuffer)

    def extractingLabel(self):
        if self.verbose:
            toPrint = "* Extracting samples and labels from dataset.."
            asteriskBuffer = ""
            for i in range(0, len(toPrint)):
                asteriskBuffer = asteriskBuffer + "*"

            print("\n" + asteriskBuffer)
            print(toPrint)
            print(asteriskBuffer)

    def labelizeInit(self):
        if self.verbose:
            toPrint = ("* Preparing to feed data into Words2Vec algorithm..")

            asteriskBuffer = ""
            for i in range(0, len(toPrint)):
                asteriskBuffer = asteriskBuffer + "*"

            print("\n" + asteriskBuffer)
            print(toPrint)
            print(asteriskBuffer)

    def words2VecInit(self):
        if self.verbose:
            toPrint = ("* Starting Words2Vec Algorithm..")

            asteriskBuffer = ""
            for i in range(0, len(toPrint)):
                asteriskBuffer = asteriskBuffer + "*"

            print("\n" + asteriskBuffer)
            print(toPrint)
            print(asteriskBuffer)

    def words2VecInit(self):
        if self.verbose:
            toPrint = ("* Creating words2Vec model and training..")

            asteriskBuffer = ""
            for i in range(0, len(toPrint)):
                asteriskBuffer = asteriskBuffer + "*"

            print(asteriskBuffer)
            print(toPrint)
            print(asteriskBuffer)

    def words2VecGraphInit(self):
        toPrint = ("* Vector Map Result from Words2Vec Algorithm:")

        asteriskBuffer = ""
        for i in range(0, len(toPrint)):
            asteriskBuffer = asteriskBuffer + "*"

        print(asteriskBuffer)
        print(toPrint)
        print(asteriskBuffer)

    def tfidfInit(self):
        if self.verbose:
            toPrint = ("* Creating TF-IDF Matrix..")

            asteriskBuffer = ""
            for i in range(0, len(toPrint)):
                asteriskBuffer = asteriskBuffer + "*"

            print(asteriskBuffer)
            print(toPrint)
            print(asteriskBuffer)

    def tfidfApply(self):
        if self.verbose:
            toPrint = ("* Using TF-IDF to reconstruct whole tweets..")

            asteriskBuffer = ""
            for i in range(0, len(toPrint)):
                asteriskBuffer = asteriskBuffer + "*"

            print(asteriskBuffer)
            print(toPrint)
            print(asteriskBuffer)

    def scaleInit(self):
        if self.verbose:
            toPrint = ("* Scaling test and train data to have 0 mean and standard deviation..")

            asteriskBuffer = ""
            for i in range(0, len(toPrint)):
                asteriskBuffer = asteriskBuffer + "*"

            print(asteriskBuffer)
            print(toPrint)
            print(asteriskBuffer)

    def neuralNetworkInit(self):
        if self.verbose:
            toPrint = ("* Creating Neural Network..")

            asteriskBuffer = ""
            for i in range(0, len(toPrint)):
                asteriskBuffer = asteriskBuffer + "*"

            print(asteriskBuffer)
            print(toPrint)
            print(asteriskBuffer)

    def neuralNetworkTrain(self):
        if self.verbose:
            toPrint = ("* Training Neural Network..")

            asteriskBuffer = ""
            for i in range(0, len(toPrint)):
                asteriskBuffer = asteriskBuffer + "*"

            print(asteriskBuffer)
            print(toPrint)
            print(asteriskBuffer)

    def done(self):
        if self.verbose:
            print("\t..done.\n")

    def printTable(self, data):
        if (self.verboseVerbose):
            if (len(data) < 5):
                print(data)
            else:
                print(data.head(5))

    def printSamplesLabelsTable(self, samples, labels):
        if (self.verboseVerbose):
            print("LABEL\tSAMPLE")
            if (len(labels) < 5):
                for i in range(0, len(labels)):
                    print(labels[i]+"\t:"+samples[i])
            else:
                for i in range(0, 5):
                    print(str(labels[i])+"\t:"+samples[i])

    def graphWords2VecMap(self, w2v, vocabKeys):
        # defining the chart
        plot_tfidf = bp.figure(plot_width=700, plot_height=600,
                               title="Wordmap Results from Word2Vec Algorithm",
                               tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                               x_axis_type=None, y_axis_type=None,
                               min_border=1)

        # list of word vectors
        word_vectors = [w2v[w] for w in tqdm(vocabKeys)]

        # Vectors -> 2D Vectors
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
        tsne_w2v = tsne_model.fit_transform(word_vectors)

        # Dataframe assignment
        tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
        tsne_df['words'] = w2v.wv.vocab.keys()[:5000]

        plot_tfidf.scatter(x='x', y='y', source=tsne_df)
        hover = plot_tfidf.select(dict(type=HoverTool))
        hover.tooltips = {"word": "@words"}
        show(plot_tfidf)

    def neuralNetworkResults(self):
        toPrint = ("* Neural Network Results:")

        asteriskBuffer = ""
        for i in range(0, len(toPrint)):
            asteriskBuffer = asteriskBuffer + "*"

        print(asteriskBuffer)
        print(toPrint)
        print(asteriskBuffer)

    def printVerbose(self, text):
        if (self.verbose):
            print(text)

    def printVerboseVerbose(self, text):
        if (self.verboseVerbose):
            print(text)
