import pandas as pd
import numpy as np
from plotly.offline import plot
from pprint import pprint
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import copy
import collections
class KMeans:

    def __init__(self, df):

        self.dataframe=df
        self.y=df[' brand']
        self.y=pd.get_dummies(self.y)
        # self.y=self.y.as_matrix(columns=None)
        self.dataframe=self.preprocess(self.dataframe)
        self.k=self.selectK(df)
        # self.centroids= dict()
        self.X = self.dataframe
        # print self.X
        # exit()
        # print clumnList
        # exit()
        rowList= range(len(self.dataframe))
        x= list()
        y=list()
        for i in range(self.k):
            # continue
            row= np.random.choice(rowList,replace=False)
            x.append(self.dataframe.ix[row])
            y.append(self.y.ix[row])
        # exit()
        # pprint(x)
        # pprint(y)
        # exit()
        self.centroids=dict()
        self.centroids['x']=x
        self.centroids['y']=y
        self.colorMap=list()
        for i in range(self.k):
            self.colorMap.append('rgba('+str(np.random.randint(0,255))+', '+str(np.random.randint(0,255))+', '+
                                 str(np.random.randint(0,255))+', .9)')
        self.traceCentroid=list()
        for i in range(self.k):
            l=list()
            # for j in range(len(self.centroids['x'])):
            dimensions= self.centroids['x'][i].index.array
            data=self.centroids['x'][i].values
            for j in range(len(dimensions)):
                l.append(dict(label=dimensions[j], values=data[j]))
            pprint(l)
            # for i in dimensions:
            #     print i
            self.traceCentroid.append(go.Splom(dimensions=l),marker=dict(color=[self.colorMap[i]],size=10,line=dict(width=0.5,
                                      color='rgb(230,230,230)')))
        # exit()
    def assign(self):
        for i in range(len(self.centroids['x'])):
            self.dataframe['distance from centroid'+str(i)]=np.sqrt((self.dataframe['x']-self.centroids['x'][i])**2+
                                                                    (self.dataframe['y']-self.centroids['y'][i])**2)
        centroid_distance_cols = ['distance from centroid{}'.format(i) for i in range(len(self.centroids['x']))]
        self.dataframe['closest'] = self.dataframe.loc[:, centroid_distance_cols].idxmin(axis=1)
        self.dataframe['closest'] = self.dataframe['closest'].map(lambda x: int(x.lstrip('distance from centroid')))
        print self.dataframe.head()
    def updateCentroids(self):
        # oldCentroids=copy.deepcopy(self.centroids)
        for i in range(len(self.centroids['x'])):
            self.centroids['x'][i]= np.mean(self.dataframe[self.dataframe['closest'] == i]['x'])
            self.centroids['y'][i]= np.mean(self.dataframe[self.dataframe['closest']==i]['y'])
        # pprint(oldCentroids)
        pprint(self.centroids)
    def cluster(self):
        self.assign()
        oldCluster= self.dataframe['closest'].copy(deep=True)
        i=0
        while not oldCluster.equals(self.dataframe['closest']) and i<300:
            self.updateCentroids()
            self.assign()
            oldCluster= self.dataframe['closest'].copy(deep=True)
            i+=1
    def plotUnClusteredGraph(self):
        l=list()
        for i in self.dataframe.columns:
            l.append(dict(label=i,values=self.dataframe[i]))
        tracePoints = go.Splom()
        data = [tracePoints]
        data.extend(self.traceCentroid)
        plot(data, filename='unclustered.html')
    def plotAssigned(self):
        tracePointsDict= dict()
        for i in range(len(self.dataframe['x'])):
            if self.dataframe['closest'][i] not in tracePointsDict:
                tracePointsDict[self.dataframe['closest'][i]]={'x':[self.dataframe['x'][i]],'y':[self.dataframe['y'][i]]}
            else:
                tracePointsDict[self.dataframe['closest'][i]]['x'].append(self.dataframe['x'][i])
                tracePointsDict[self.dataframe['closest'][i]]['y'].append(self.dataframe['y'][i])
        # pprint(tracePointsDict)
        tracePoints=list()
        for i in tracePointsDict.keys():
            tracePoints.append(go.Scatter(x=tracePointsDict[i]['x'],y=tracePointsDict[i]['y'],name='Closest From Centroid '+ str(i),
                                          mode= 'markers',marker=dict(size=10,
                                                             color=self.colorMap[i],
                                                             line= dict(
                                                                 width=2,
                                                                 color='rgb(0, 0, 0)'
                                                             ))))
        pprint(tracePoints)
        #     tracePoints.append(go.Scatter(x=[self.dataframe['x'][i]],y=[self.dataframe['y'][i]],name= str(self.dataframe['closest'][i]),
        #                                   mode= 'markers',
        #                                          marker=dict(size=10,
        #                                                      color=self.dataframe['color'][i],
        #                                                      line= dict(
        #                                                          width=2,
        #                                                          color='rgb(0, 0, 0)'
        #                                                      ))
        #                                   ))
        # data = tracePoints
        data= list()
        data.extend(tracePoints)
        data.extend(self.traceCentroid)
        plot(data, filename='assigned.html')
    @staticmethod
    def selectK(dataframe):
        l= dataframe[' brand'].unique()
        return len(l)
        # exit()
    @staticmethod
    def preprocess(df):
        df= KMeans.getDummyVariables(df)
        df= KMeans.reduceDimensionality(df)
        return df
    @staticmethod
    def getDummyVariables(df):
        dummiesCylinders=pd.get_dummies(df[' cylinders'],prefix='cylinders',dummy_na=True)
        # print dummiesCylinders.head()
        dummiesYear= pd.get_dummies(df[' year'],prefix='year',dummy_na=True)
        df= pd.concat([df,dummiesCylinders,dummiesYear],axis=1)
        df= df.drop(' cylinders',axis=1)
        df= df.drop(' year',axis=1)
        # print df.columns
        df[' weightlbs']= df[' weightlbs'].replace(r'\s', -1, regex=True)
        df[' cubicinches'] = df[' cubicinches'].replace(r'\s', -1, regex=True)
        # print df[' weightlbs']
        # exit()
        df[' cubicinches']= df[' cubicinches'].astype(str).astype(int)
        df[' weightlbs']= df[' weightlbs'].astype(str).astype(int)
        # pd.to_numeric(df[' weightlbs']).astype(int)
        df= df.replace(-1,np.nan, regex=True)
        # print df.dtypes
        # exit()
        return df
        # df=df.drop(' brand',axis=1)
        # print df.head()
        # exit()
    @staticmethod
    def reduceDimensionality(df):
        model = RandomForestRegressor(random_state=1, max_depth=10)
        brandColumn= df[' brand']
        brandColumn= pd.get_dummies(brandColumn)
        # print 'BrandColumn'
        print brandColumn.head()
        # exit()
        df = df.drop(' brand', axis=1)
        # print df.head()
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp = imp.fit(df)
        df_imp = imp.transform(df)
        # exit()
        model.fit(df_imp,brandColumn)
        features = df.columns
        importance = model.feature_importances_
        sortDf=pd.DataFrame({'features':df.columns.get_values(),'importance':importance})
        sortDf.sort_values(['importance'],axis=0, ascending=False, inplace=True)
        importantFeatures=list()
        for i in range(len(sortDf['features'])):
            if sortDf['importance'][i]>=0.05:
                importantFeatures.append(sortDf['features'][i])
        # pprint(importantFeatures)
        # importances.sort()
        # list_sort= zip(features,importances)
        # pprint(list_sort)
        # features.sort(key=importances)
        # print sortDf.head()
        # pprint(importances)
        l= df.columns.get_values()
        for i in l:
            if i not in importantFeatures:
                df= df.drop(i,axis=1)
        # exit()
        return df
# df = pd.DataFrame({
#     'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
#     'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
# })
df= pd.read_csv('cars.csv')
if __name__== '__main__':
    o=KMeans(df)
    o.plotUnClusteredGraph()
    o.cluster()
    # o.assign()
    o.plotAssigned()
    # o.updateCentroids()