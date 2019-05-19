import pandas as pd
from multiprocessing.pool import ThreadPool
from scipy.spatial import distance
import pickle

#getting the entire dataframe in a variable df

df = pd.read_csv("creditcard.csv")
#given a column returns the min and max value in that column
def calEdge(col):
  min = df.loc[0,col]
  max = df.loc[0,col]
  for item in df.loc[:,col].values:
    if item < min:
      min = item
    if item > max:
      max = item
  return min,max

#normalises the database
def normalise():
  for item in list(df):
    print(item)
    min,max = calEdge(item)
    for element in range(284807):
      df.loc[element,item] = df.loc[element,item] - min
      df.loc[element,item] = df.loc[element,item]/(max-min)

normalise()
df.to_pickle("./normalized.pkl")

dist = {}

def calcDist(row1,row2):
  a = list(df.loc[row2,:])
  b = list(df.loc[row1,:])
  distance = 0
  for item in range(0,len(a)-1):
    x = (a[item] - b[item])**2
    x = x/len(a)
    distance = distance + x
  dist[(row1,row2)] = distance
  return distance

item = []
for row1 in range(284807):
  for row2 in range(284807):
    itemElement = [row1,row2]
    item.append(itemElement)

pool = ThreadPool()
pool.map(lambda x: calcDist(x), item)

f = open("dist.pkl","wb")
pickle.dump(dist,f)
f.close()