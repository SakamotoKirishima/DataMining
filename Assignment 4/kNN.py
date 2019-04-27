#!/usr/bin/env python
# coding: utf-8

from optparse import OptionParser
import numpy as np
import operator
from math import sqrt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from Preprocess import preprocess


def getEuclidianDistance(inst1, inst2):
    sum = 0
    for i in range(0, len(inst1)):
        sum = sum + (inst1[i] - inst2[i]) ** 2
    return sqrt(sum)


class KNearestNeighbors:
    def __init__(self, k):
        self.k = k
        self.test_X= self.train_X= self.test_Y=self.train_Y= self.predictions= None

    def getTrainTest(self, X, Y):
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(X, Y, test_size=0.3)

    def getMaxCount(self, nearest):
        count = defaultdict(int)
        labels = self.train_Y[nearest]
        for i in labels:
            count[i] = count[i] + 1
        return max(count.iteritems(), key=operator.itemgetter(1))[0]

    def predict(self, inst):
        dist = np.array([])
        for i in range(0, len(self.train_X)):
            dist = np.append(dist, getEuclidianDistance(self.train_X[i], inst))
        sortIndex = np.argsort(dist)
        prediction = self.getMaxCount(sortIndex[0:self.k])
        return prediction

    def getPredictions(self):
        finalPredictions = np.array([])
        for i in range(0, len(self.test_X)):
            finalPredictions = np.append(finalPredictions, self.predict(self.test_X[i]))
        c = 0
        self.predictions = finalPredictions
        for i in range(0, len(self.test_X)):
            if finalPredictions[i] == self.test_Y[i]:
                c = c + 1
        accuracy = float(c) / len(self.test_Y)
        return accuracy


if __name__ == "__main__":
    ## input parameter reading
    # optparser = OptionParser()
    # optparser.add_option('-k', '--knear',
    #                      dest='knear',
    #                      help='filename containing csv',
    #                      default='5')
    # (options, args) = optparser.parse_args()
    data = preprocess()
    X = data[:, 0:8]
    Y = data[:, 8]
    k = 5
    # print k
    knn = KNearestNeighbors(k)
    knn.getTrainTest(X, Y)

    acc = knn.getPredictions()
    print acc
    confusionMatrix = defaultdict(lambda: defaultdict(int))
    for i in range(0, len(knn.test_X)):
        confusionMatrix[knn.test_Y[i]][knn.predictions[i]] = confusionMatrix[knn.test_Y[i]][knn.predictions[i]] + 1
    for key in confusionMatrix:
        print key, confusionMatrix[key]