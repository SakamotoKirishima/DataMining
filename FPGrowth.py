class Node:
    def __init__(self, name, noOfOccurrences, parent):
        self.name = name
        self.noOfOccurences = noOfOccurrences
        self.parent = parent
        self.children = dict()
        self.nodeLink = None

    def addOccurences(self, noOfMoreOccurences):
        self.noOfOccurences += noOfMoreOccurences

    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.noOfOccurences)
        for child in self.children.values():
            child.disp(ind + 1)


def makeTree(dataset, minsupport=1):
    headerTable = dict()
    for row in dataset:
        for element in row:
            headerTable[element] = headerTable.get(element, 0) + dataset[row]
    keysToDelete = list()
    for key in headerTable.keys():
        if headerTable[key] < minsupport:
            keysToDelete.append(key)
    for key in keysToDelete:
        del headerTable[key]
    itemSet = set(headerTable.keys())
    for key in headerTable.keys():
        headerTable[key] = [headerTable[key], None]
    tree = Node('NULL', 1, None)
    for row, count in dataset.items():
        localDict = dict()
        for element in row:
            if element in itemSet:
                localDict[element] = headerTable[element][0]
            if len(localDict) > 0:
                orderedItems = [v[0] for v in sorted(localDict.items(), key=lambda p: p[1], reverse=True)]
                updateTree(orderedItems, tree, headerTable, count)
    return tree, headerTable


def updateTree(items, tree, headerTable, count):
    if items[0] in tree.children:
        tree.children[items[0]].addOccurences(count)
    else:
        tree.children[items[0]] = Node(items[0], count, tree)
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = tree.children[items[0]]
        else:
            updateHeaderTable(headerTable[items[0]][1], tree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], tree.children[items[0]], headerTable, count)


def updateHeaderTable(node, targetNode):
    while node.nodeLink is not None:
        node = node.nodeLink
    node.nodeLink = targetNode


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def addPath(node, path):
    if node.parent is not None:
        path.append(node.name)
        addPath(node.parent, path)


def getPaths(basePath, node):
    allPaths = dict()
    while node is not None:
        path = list()
        addPath(node, path)
        if len(path) > 1:
            allPaths[frozenset(path[1:])] = node.noOfOccurences
        node = node.nodeLink
    return allPaths


if __name__ == '__main__':
    simpDat = loadSimpDat()
    print(simpDat)
    initSet = createInitSet(simpDat)
    print(initSet)
    myFPtree, myHeaderTab = makeTree(initSet, 3)
    # myFPtree.disp()
    path = getPaths('x', myHeaderTab['x'][1])
