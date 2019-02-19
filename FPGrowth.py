import argparse
from Preprocessing import Import


class Node:
    """
    Class to store the values required to identify each node
    """

    def __init__(self, value=tuple(), noOfOccurrences=1, parent=None):
        """
        __init__ method
        :param value: A tuple containing the attribute-value pair
        :param noOfOccurrences: no of occurrences of the attribute-value pair
        :param parent: parent of Node
        """
        self.value = value
        self.noOfOccurences = noOfOccurrences
        self.parent = parent
        self.children = dict()
        self.nodeLink = None

    def linkNode(self):
        """
        Method to find the linking node
        :return:
        """
        node = self
        while node.nodeLink is not None:
            node = node.nodeLink
        return node


class Tree:
    """
    Class to store the information of the FP tree and mine it
    """

    def __init__(self):
        """
        __init__ method
        """
        self.headerTable = dict()
        self.tree = Node()
        self.allPaths = dict()

    def makeTree(self, dataSet):
        """
        Method to create the FP tree
        :param dataSet: the data set to find frequent item sets in
        :return: None
        """
        minSup = 1
        for row in dataSet:
            for element in row:
                self.headerTable[element] = self.headerTable.get(element, 0) + dataSet[row]
        keysToDelete = list()
        for key in self.headerTable.keys():
            if self.headerTable[key] < minSup:
                keysToDelete.append(key)
        for key in keysToDelete:
            self.headerTable.pop(key)
        freq_item_set = list()
        for key in self.headerTable.keys():
            if key not in freq_item_set:
                freq_item_set.append(key)
        if len(freq_item_set) == 0:
            return
        for key in self.headerTable:
            self.headerTable[key] = [self.headerTable[key], None]
        self.tree = Node(('Null Set',), 1, None)
        for row, count in dataSet.items():
            localDict = dict()
            for item in row:
                if item in freq_item_set:
                    localDict[item] = self.headerTable[item][0]
            if len(localDict) > 0:
                orderedItems = [v[0] for v in sorted(localDict.items(), key=lambda p: p[1], reverse=True)]
                self.updateTree(self.tree, self.headerTable, orderedItems, count)

    def getPaths(self, node):
        """
        Method to get conditional pattern
        :param node: Node to draw path from
        :return: None
        """
        check = True
        while check:
            prefixPath = list()
            self.addPath(node, prefixPath)
            if len(prefixPath) > 1:
                tempTuple = list()
                for i in range(1, len(prefixPath)):
                    if prefixPath[i] not in tempTuple:
                        tempTuple.append(prefixPath[i])
                self.allPaths[tuple(tempTuple)] = node.noOfOccurences
            node = node.nodeLink
            if node is None:
                check = False

    @staticmethod
    def updateTree(tree, table, items, count):
        """
        Static method called by make Tree to update the add nodes and modify the tree
        :param tree: FP tree
        :param table: containing what is in the dataset and how often it occurs
        :param items: items to put into the FP tree
        :param count: no of occurrences of an item
        :return: None
        """
        if items[0] not in tree.children:
            tree.children[items[0]] = Node(items[0], count, tree)
            if table[items[0]][1] is not None:
                node = table[items[0]][1].linkNode()
                node.nodeLink = tree.children[items[0]]

            else:
                table[items[0]][1] = tree.children[items[0]]

        else:
            tree.children[items[0]].noOfOccurences += count
        if len(items) <= 1:
            return
        Tree.updateTree(tree.children[items[0]], table, items[1::], count)

    @staticmethod
    def addPath(node, path):
        """
        Static method to find a path from node to root recursively
        :param node: input node
        :param path: path found
        :return:
        """
        if node.parent is None:
            return
        path.append(node.value)
        Tree.addPath(node.parent, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attribute')
    parser.add_argument('value')
    args = parser.parse_args()
    value = args.value
    if args.value.isdigit():
        value = int(args.value)
    data = Import.import_data()
    treeObj = Tree()
    treeObj.makeTree(data)
    try:
        pathValue = treeObj.headerTable[(args.attribute, value)]
    except KeyError:
        print('Value not found')
        exit(1)
    treeObj.getPaths(pathValue[1])
    print(treeObj.allPaths)
    stringToAdd = ''
    for row in treeObj.allPaths:
        for i in range(len(row)):
            phraseToAdd = str(row[i][0]) + '=' + str(row[i][1])
            stringToAdd += (phraseToAdd)

            if i < len(row) - 1:
                stringToAdd += ','
            else:
                stringToAdd += ' '
        stringToAdd += '(' + str(treeObj.allPaths[row]) + ')' + '\n'
    fo = open('Freq_Items_sup.txt', 'w')
    fo.write(stringToAdd)
