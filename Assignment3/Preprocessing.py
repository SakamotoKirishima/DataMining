import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# print 1


def preprocess():
    df = pd.read_csv('creditcard.csv')
    # print 2
    df1 = df.drop(u'Time', axis=1)
    # print 3
    model = RandomForestRegressor(random_state=1, max_depth=10)
    # print 4
    model.fit(df1, df.Time)
    # print 5
    features = df.columns
    importances = model.feature_importances_
    # print 6
    indices = np.argsort(importances)[-2:]
    # print 7
    importantFeatures = list()
    for i in indices:
        importantFeatures.append(features[i])
    for i in df.columns:
        if i not in importantFeatures:
            df = df.drop(i, axis=1)

    # x = list(reader)
    df = df.truncate(before=1, after=10000)
    array = df.values
    result = np.asmatrix(array)
    # result= numpy.reshape(result,(10000,10))
    # print result.shape
    fo = open('creditcard.dat', 'wb')
    pickle.dump(result, fo)
    fo.close()
