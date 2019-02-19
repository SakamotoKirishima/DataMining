from Preprocessing import Import
import argparse


class Node:
    def __init__(self, value=tuple(), noOfOccurrences=1, parent=None):
        self.value = value
        self.noOfOccurences = noOfOccurrences
        self.parent = parent
        self.children = dict()
        self.nodeLink = None


def makeTree(dataSet):
    minSupport = 1
    headerTable = dict()
    for row in dataSet:
        for element in row:
            headerTable[element] = headerTable.get(element, 0) + dataSet[row]
    keysToDelete = list()
    for key in headerTable.keys():
        if headerTable[key] < minSupport:
            keysToDelete.append(key)
    for key in keysToDelete:
        del headerTable[key]
    itemSet = set(headerTable.keys())
    for key in headerTable.keys():
        headerTable[key] = [headerTable[key], None]
    tree = Node()
    for row, count in dataSet.items():
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
        tree.children[items[0]].noOfOccurences += count
    else:
        tree.children[items[0]] = Node(items[0], count, tree)
        if headerTable[items[0]][1] is not None:
            node= findNodeLink(headerTable[items[0]][1])
            node.nodelink= tree.children[items[0]]

        else:
            headerTable[items[0]][1] = tree.children[items[0]]
    if len(items) <= 1:
        pass
    else:
        updateTree(items[1::], tree.children[items[0]], headerTable, count)


def findNodeLink(node):
    while node.nodeLink is not None:
        node = node.nodeLink
    return node


def addPath(node, path):
    if node.parent is None:
        return
    path.append(node.value)
    addPath(node.parent, path)


def getPaths(node):
    allPaths = dict()
    check = True
    while check:
        path = list()
        addPath(node, path)
        if len(path) > 1:
            tempTuple=list()
            for i in range(1,len(path)):
                if path[i] not in tempTuple:
                    tempTuple.append(path[i])
            allPaths[tuple(tempTuple)] = node.noOfOccurences
        node = node.nodeLink
        if node is None:
            check= False
    return allPaths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attribute')
    parser.add_argument('value')
    args = parser.parse_args()
    value = args.value
    if args.value.isdigit():
        value = int(args.value)
    data = Import.import_data()
    fpTree, headerTab = makeTree(data)
    try:
        pathValue = headerTab[(args.attribute, value)]
    except KeyError:
        print('Value not found')
        exit(1)
    path = getPaths(pathValue[1])
    stringToAdd = ''
    for row in path:
        for i in range(len(row)):
            phraseToAdd = str(row[i][0]) + '=' + str(row[i][1])
            stringToAdd += (phraseToAdd)

            if i < len(row) - 1:
                stringToAdd += ','
            else:
                stringToAdd += ' '
        stringToAdd += '(' + str(path[row]) + ')' + '\n'
    fo = open('Freq_Items_sup.txt', 'w')
    fo.write(stringToAdd)
