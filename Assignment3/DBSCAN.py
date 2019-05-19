import cPickle as pickle
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
from Preprocessing import preprocess


class Status:
    """
    Class to determine the status of each point
    """
    def __init__(self):
        """
        __init__() method
        """
        self.status = 'new'

    def setVisited(self):
        """
        Method to set visited status
        """
        self.status = 'visited'

    def setNoise(self):
        """
        Method to set status as noise
        :return:
        """
        self.status = 'noise'

    def getStatus(self):
        """
        MEthod to get status
        :return: status
        """
        return self.status


class Clustering:
    """
    Class that performs clustering
    """
    status = None
    isMember = None
    distance = None
    distanceIdx = None

    def __init__(self, points, eps=0.15, minSup=2):
        """
        __init__ method
        :param points: points input
        :param eps: minimum distance(epsilon)
        :param minSup: minimum number of points
        """
        self.points = points
        self.eps = eps
        self.minSup = minSup

    def mainDBSCAN(self):
        """
        Main DBSCAN method
        :return: all clusters
        """
        clusters = list()
        nPts = self.points.shape[0]
        self.computeDistance()

        for k in range(nPts):
            if Clustering.status[k].getStatus() == 'visited':
                continue
            Clustering.status[k].setVisited()
            neighbors = self.regionQuery(k)
            if len(neighbors) < self.minSup:
                self.status[k].setNoise()
            else:
                cluster = self.expandCluster(k, neighbors)
                clusters.append(cluster)

        return clusters

    def expandCluster(self, point, neighbors):
        """
        Method to expand a cluster
        :param point: point input
        :param neighbors: neighbours
        :return: expanded cluster
        """

        cluster = list()
        cluster = self.addToCluster(cluster, point)

        while neighbors:
            k = neighbors.pop()
            if Clustering.status[k].getStatus() != 'visited':
                Clustering.status[k].setVisited()
                extendedNeighbors = self.regionQuery(k)
                if len(extendedNeighbors) >= self.minSup:
                    neighbors.update(extendedNeighbors)
            if not self.isMember[k]:
                self.addToCluster(cluster, k)

        return cluster

    def regionQuery(self, center):
        """
        Query to find the nearest points
        :param center: the point input
        :return: neighbors
        """
        neighbors = set()
        i = 0
        while Clustering.distance[center, Clustering.distanceIdx[center, i]] <= self.eps:
            neighbors.add(Clustering.distanceIdx[center, i])
            i += 1
        return neighbors

    def computeDistance(self):
        """
        Calculate the distance between all the points
        :return:
        """

        nPts = self.points.shape[0]

        Clustering.distance = np.zeros((nPts, nPts))
        Clustering.isMember = [False] * nPts
        Clustering.status = list()
        for i in range(nPts):
            Clustering.status.append(Status())
        # print(Clustering.status)
        n = self.points.sum(axis=1).dot(np.ones((1, nPts))).A
        nIntersect = self.points.dot(self.points.transpose())
        Clustering.distance = 1.0 - (nIntersect / (n + n.T - nIntersect))
        Clustering.distanceIdx = np.argsort(Clustering.distance, axis=1)

    @staticmethod
    def addToCluster(cluster, k):
        """
        Add a point to a cluster
        :param cluster: cluster input
        :param k: the point input
        :return: expanded cluster
        """
        cluster.append(k)
        Clustering.isMember[k] = True
        return cluster


def plotUnclustered(points):
    """
    Function to plot unclustered graph
    :param points: points input
    :return:
    """
    pointArray = points.A
    x = pointArray[:, 0]
    y = pointArray[:, 1]
    trace = go.Scatter(
        x=x,
        y=y,
        name='Points',
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(61, 5, 169, .8)',
            line=dict(
                width=2,
                color='rgb(0, 0, 0)'
            )
        )
    )

    data = [trace]
    plot(data, filename='unclustered.html')


def plotClusters(points, clusters, noise):
    """
    Plot the clusters
    :param points: points input
    :param clusters: clusters
    :param noise: noise points
    :return:
    """
    tracePoints = list()
    j = 0
    for cluster in clusters:
        trace = dict()
        trace['x'] = list()
        trace['y'] = list()
        for i in cluster:
            # print(points.A[0])
            trace['x'].append(points.A[i][0])
            trace['y'].append(points.A[i][1])
        tracePoints.append(go.Scatter(
            x=trace['x'],
            y=trace['y'],
            name='Cluster' + str(j),
            mode='markers',
            marker=dict(
                size=10,
                color='rgba(' + str(np.random.randint(0, 255)) + ', ' + str(np.random.randint(0, 255)) + ', ' + str(
                    np.random.randint(0, 255)) + ', .8)',
                line=dict(
                    width=2,
                    color='rgb(0, 0, 0)'
                )
            )
        ))
        j+=1

    trace = dict()
    trace['x'] = list()
    trace['y'] = list()
    for i in noise:
        trace['x'].append(points.A[i][0])
        trace['y'].append(points.A[i][1])
    tracePoints.append(go.Scatter(
        x=trace['x'],
        y=trace['y'],
        name='Noise',
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(' + str(np.random.randint(0, 255)) + ', ' + str(np.random.randint(0, 255)) + ', ' + str(
                np.random.randint(0, 255)) + ', .8)',
            line=dict(
                width=2,
                color='rgb(0, 0, 0)'
            )
        )
    ))
    data= tracePoints
    plot(data, filename='clustered.html')

if __name__ == "__main__":
    preprocess()
    file_descriptor = open("creditcard.dat", "r")
    pointsToCluster = pickle.load(file_descriptor)
    # plotUnclustered(pointsToCluster)
    # pointsToCluster= np.asmatrix(pointsToCluster)
    #     pointsToCluster= np.matrix('''0	 -1.359807134 -0.072781173	2.536346738	1.378155224	-0.33832077	0.462387778	0.239598554	0.098697901	0.36378697;
    # 0 1.191857111 0.266150712 0.166480113 0.448154078 0.060017649 -0.082360809 -0.078802983 0.085101655 -0.255425128;
    # 1 -1.358354062 -1.340163075 1.773209343 0.379779593 -0.503198133 1.800499381 0.791460956 0.247675787 -1.514654323;
    # 1 -0.966271712 -0.185226008 1.79299334 -0.863291275 -0.01030888 1.247203168 0.23760894 0.377435875 -1.387024063;
    # 2 -1.158233093 0.877736755 1.548717847 0.403033934 -0.407193377 0.095921462 0.592940745 -0.270532677 0.817739308;
    # 2 -0.425965884 0.960523045 1.141109342 -0.16825208 0.420986881	 -0.029727552 0.476200949 0.260314333 -0.568671376;
    # 4 1.229657635 0.141003507 0.045370774 1.202612737 0.191880989 0.272708123 -0.005159003 0.08121294 0.464959995;
    # 7 -0.644269442 1.417963545 1.074380376 -0.492199018 0.948934095 0.428118463 1.120631358 -3.807864239 0.615374731;
    # 7 -0.894286082 0.286157196 -0.113192213 -0.27152613 2.66959866 3.721818061 0.370145128 0.851084443 -0.392047587;
    # 9 -0.338261752 1.119593376 1.044366552 -0.222187277 0.499360806 -0.246761101 0.651583206 0.069538587 -0.736727316
    # ''')
    # print pointsToCluster
    # exit()

    # print(type(pointsToCluster))
    clusterOb = Clustering(pointsToCluster, 0.001, 2)
    clusterOb.computeDistance()
    clusters = clusterOb.mainDBSCAN()
    if len(clusters) > 0:
        print "The following clusters were found:"
        cluster_lengths = []
        for cluster in clusters:
            print cluster
            cluster_lengths.append(len(cluster))
        noise = list()
        for point in range(len(Clustering.status)):
            if clusterOb.status[point].getStatus() == 'noise':
                noise.append(point)
        print "\nThe noise points are:\n", noise
        # plotClusters(pointsToCluster, clusters, noise)
        print len(clusters[0])/10000
    else:
        print "No clusters were found!"
