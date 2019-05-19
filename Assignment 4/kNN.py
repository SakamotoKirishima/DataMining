import numpy as np
import operator
from math import sqrt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from Preprocess import preprocess


def getEuclidianDistance(instance1, instance2):
    """
    Get distance between 2 instances
    :param instance1: First point
    :param instance2: Second point
    :return:
    """
    sum = 0
    for i in range(0, len(instance1)):
        sum = sum + (instance1[i] - instance2[i]) ** 2
    return sqrt(sum)


class KNearestNeighbors:
    """
    Class to implement K Nearest Neighbors
    """
    def __init__(self, k, nClases):
        """
        __init__() method
        :param k: k value
        :param nClases: no of classes
        """
        self.k = k
        self.test_X= self.train_X= self.test_Y=self.train_Y= self.predictions= None
        self.nclasses=nClases
    def getTrainTest(self, X, Y):
        """
        Get train and test sets
        :param X: X attributes
        :param Y: Y attributes
        :return: train and test models
        """
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(X, Y, test_size=0.3)

    def getMaxCount(self, nearest):
        """
        Get Maximum Count
        :param nearest: nearest neighbors
        :return: maximum voted value
        """
        count = defaultdict(int)
        labels = self.train_Y[nearest]
        for i in labels:
            count[i] = count[i] + 1
        return max(count.iteritems(), key=operator.itemgetter(1))[0]

    def predict(self, inst):
        """
        Method to predict a value of an instance
        :param inst: Instance
        :return: predicted value
        """
        dist = np.array([])
        for i in range(0, len(self.train_X)):
            dist = np.append(dist, getEuclidianDistance(self.train_X[i], inst))
        sortIndex = np.argsort(dist)
        prediction = self.getMaxCount(sortIndex[0:self.k])
        return prediction

    def getPredictions(self):
        """
        Get all the predictions for all attributes
        :return: accuracy
        """
        finalPredictions = np.array([])
        for i in range(0, len(self.test_X)):
            finalPredictions = np.append(finalPredictions, self.predict(self.test_X[i]))
        c = 0
        tp=fp=fn=0
        self.predictions = finalPredictions
        for i in range(0, len(self.test_X)):
            if finalPredictions[i] == self.test_Y[i]:
                c = c + 1
            if finalPredictions[i] == 'spec_prior' and finalPredictions[i]== self.test_Y[i]:
                tp+=1
            elif finalPredictions[i]== 'not_recom' and finalPredictions[i] != self.test_Y[i]:
                fn+=1
            elif finalPredictions[i]== 'spec_prior' and finalPredictions[i] != self.test_Y[i]:
                fp+=1
        accuracy = float(c) / len(self.test_Y)
        print(float(tp)/float(tp+fp)) #precision
        print(float(tp)/float(tp+fn)) #recall
        return accuracy


if __name__ == "__main__":
    data = preprocess()
    X = data[:, 0:8]
    Y = data[:, 8]
    k = 4
    knn = KNearestNeighbors(5,5)
    knn.getTrainTest(X, Y)

    acc = knn.getPredictions()
    print acc
    confusionMatrix = defaultdict(lambda: defaultdict(int))
    for i in range(0, len(knn.test_X)):
        confusionMatrix[knn.test_Y[i]][knn.predictions[i]] = confusionMatrix[knn.test_Y[i]][knn.predictions[i]] + 1
    for key in confusionMatrix:
        print key, confusionMatrix[key]