import pandas as pd
import numpy as np
from plotly.offline import plot
from pprint import pprint
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


class KMeans:

    def __init__(self, df):

        self.dataframe = df
        self.dataframe.dropna()
        self.y = df[' brand']
        self.y = pd.get_dummies(self.y)
        # self.y=self.y.as_matrix(columns=None)
        self.dataframe = self.preprocess(self.dataframe)
        self.dataframe = pd.concat([self.dataframe, self.y], axis=1)
        # self.dataframe.fillna(0)
        self.k = self.selectK(df)
        # self.centroids= dict()
        self.X = self.dataframe
        # print self.X
        # exit()
        # print clumnList
        # exit()
        rowList = range(len(self.dataframe))
        x = list()
        y = list()
        for i in range(self.k):
            # continue
            row = np.random.choice(rowList, replace=False)
            x.append(self.dataframe.ix[row])
            y.append(self.y.ix[row])
        # exit()
        # pprint(x)
        # pprint(y)
        # exit()
        self.centroids = dict()
        self.centroids['x'] = x
        self.centroids['y'] = y
        self.colorMap = list()
        for i in range(self.k):
            self.colorMap.append(
                'rgba(' + str(np.random.randint(0, 255)) + ', ' + str(np.random.randint(0, 255)) + ', ' +
                str(np.random.randint(0, 255)) + ', .9)')
        self.traceCentroid = list()
        for i in range(self.k):
            l = list()
            # for j in range(len(self.centroids['x'])):
            dimensions = self.centroids['x'][i].index.array
            data = self.centroids['x'][i].values
            for j in range(len(dimensions)):
                l.append(dict(label=dimensions[j], values=[data[j]]))
            # pprint(l)
            # for i in dimensions:
            #     print i
            self.traceCentroid.append(go.Splom(dimensions=l, name='Centroid' + str(i),
                                               marker=dict(color=[self.colorMap[i]], size=10, line=dict(width=0.5,
                                                                                                        color='rgb(255,0,0)'))))
        # exit()

    def plotUnClusteredGraph(self):
        l = list()
        for i in self.dataframe.columns:
            l.append(dict(label=i, values=self.dataframe[i].tolist()))
        tracePoints = go.Splom(dimensions=l, name='Points',
                               marker=dict(color=['rgb(255,0,0)'], size=10, line=dict(width=0.5,
                                                                                          color='rgb(255,0,0)')))
        data = [tracePoints]
        data.extend(self.traceCentroid)
        plot(data, filename='unclustered.html')

    def assign(self):
        # print self.dataframe.dtypes
        # exit()
        # self.dataframe= pd.concat([self.dataframe,self.y],axis=1)
        for i in range(self.k):
            self.dataframe['distance from centroid{}'.format(i)] = 0
            for j in range(len(self.dataframe)):
                euclidianDistance = 0
                for k in self.centroids['x'][i].index:
                    # print self.dataframe[k][j],
                    euclidianDistance += np.square(self.dataframe[k][j] - self.centroids['x'][i][k])
                # print
                euclidianDistance = np.sqrt(euclidianDistance)
                # print euclidianDistance
                self.dataframe['distance from centroid{}'.format(i)][j] = euclidianDistance
        # print self.dataframe.head()
        # exit()
        centroid_distance_cols = ['distance from centroid{}'.format(i) for i in range(self.k)]
        self.dataframe['closest'] = self.dataframe.loc[:, centroid_distance_cols].idxmin(axis=1)
        # self.dataframe['closest'] = self.dataframe['closest'].map(lambda x: int(x.lstrip('distance from centroid')))
        self.dataframe['closest'] = self.dataframe.closest.str.replace('distance from centroid', '')
        # self.dataframe['closest']= self.dataframe['closest'].astype(int)
        # self.dataframe.dropna()
        # self.dataframe= self.dataframe.drop(self.dataframe[(self.dataframe['closest'] != '1') & (self.dataframe['closest'] != '2') &(self.dataframe['closest'] != '0')].index)
        # print self.dataframe.iloc[14]
        # print self.dataframe.head()

    def updateCentroids(self):
        i = 0
        # print type(self.centroids['x'][0])
        for i in range(self.k):
            meanCentroid = self.dataframe[self.dataframe['closest'] == str(i)].mean()
            # print type(self.centroids['x'])
            # print meanCentroid
            for j in self.centroids['x'][i].index:
                self.centroids['x'][i][j] = meanCentroid[j]
            # print self.centroids['x'][i]

            # print self.centroids['x'][0].index
            # name = self.centroids['x'][0].index[0]
            # print self.centroids['x'][0][name]
            # print meanCentroid[name]

    def cluster(self):
        self.assign()
        i = 0
        while True:
            closest_centroids = self.dataframe['closest'].copy(deep=True)
            self.updateCentroids()
            self.assign()
            # print i
            i += 1
            if closest_centroids.equals(self.dataframe['closest']) or i > 30:
                break

    def plotClusteredGraph(self):
        tracePoints = list()
        i = 0
        for i in range(self.k):
            l = list()
            dataframeForCentroid = self.dataframe[self.dataframe['closest'] == str(i)]
            # print dataframeForCentroid.head()
            # exit()
            # print type(dataframeForCentroid)
            for j in dataframeForCentroid.columns:
                # print type(dataframeForCentroid[j])
                l.append(dict(label=j, values=dataframeForCentroid[j]))
            tracePoints.append(go.Splom(dimensions=l, name='Closest from centroid'+str(i),
                                        marker=dict(color=[self.colorMap[int(i)]]* len(dataframeForCentroid.columns),
                                                    size=10, line=dict(width=0.5, color=self.colorMap[int(i)]))))
        data = tracePoints
        data.extend(self.traceCentroid)
        plot(data, filename='clustered.html')

    @staticmethod
    def selectK(dataframe):
        l = dataframe[' brand'].unique()
        return len(l)
        # exit()

    @staticmethod
    def preprocess(df):
        df = KMeans.getDummyVariables(df)
        df = KMeans.reduceDimensionality(df)
        return df

    @staticmethod
    def getDummyVariables(df):
        dummiesCylinders = pd.get_dummies(df[' cylinders'], prefix='cylinders', dummy_na=True)
        # print dummiesCylinders.head()
        dummiesYear = pd.get_dummies(df[' year'], prefix='year', dummy_na=True)
        df = pd.concat([df, dummiesCylinders, dummiesYear], axis=1)
        df = df.drop(' cylinders', axis=1)
        df = df.drop(' year', axis=1)
        # print df.columns
        df[' weightlbs'] = df[' weightlbs'].replace(r'\s', -1, regex=True)
        df[' cubicinches'] = df[' cubicinches'].replace(r'\s', -1, regex=True)
        # print df[' weightlbs']
        # exit()
        df[' cubicinches'] = df[' cubicinches'].astype(str).astype(int)
        df[' weightlbs'] = df[' weightlbs'].astype(str).astype(int)
        # pd.to_numeric(df[' weightlbs']).astype(int)
        df = df.replace(-1, np.nan, regex=True)
        # print df.dtypes
        # exit()
        return df
        # df=df.drop(' brand',axis=1)
        # print df.head()
        # exit()

    @staticmethod
    def reduceDimensionality(df):
        model = RandomForestRegressor(random_state=1, max_depth=10)
        brandColumn = df[' brand']
        brandColumn = pd.get_dummies(brandColumn)
        # print 'BrandColumn'
        # print brandColumn.head()
        # exit()
        df = df.drop(' brand', axis=1)
        # print df.head()
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp = imp.fit(df)
        df_imp = imp.transform(df)
        # exit()
        model.fit(df_imp, brandColumn)
        features = df.columns
        importance = model.feature_importances_
        sortDf = pd.DataFrame({'features': df.columns.get_values(), 'importance': importance})
        sortDf.sort_values(['importance'], axis=0, ascending=False, inplace=True)
        importantFeatures = list()
        for i in range(len(sortDf['features'])):
            if sortDf['importance'][i] >= 0.05:
                importantFeatures.append(sortDf['features'][i])
        # pprint(importantFeatures)
        # importances.sort()
        # list_sort= zip(features,importances)
        # pprint(list_sort)
        # features.sort(key=importances)
        # print sortDf.head()
        # pprint(importances)
        l = df.columns.get_values()
        for i in l:
            if i not in importantFeatures:
                df = df.drop(i, axis=1)
        # exit()
        return df


df = pd.read_csv('cars.csv')
if __name__ == '__main__':
    o = KMeans(df)
    o.plotUnClusteredGraph()
    o.cluster()
    o.plotClusteredGraph()
    # o.assign()
    # o.plotAssigned()
    # o.updateCentroids()
