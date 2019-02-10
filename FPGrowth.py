class Node:
    def __init__(self,name,noOfOccurrences,parent):
        self.name= name
        self.noOfOccurences=noOfOccurrences
        self.parent= parent
        self.children= dict()
    def addOccurences(self, noOfMoreOccurences):
        self.noOfOccurences+= noOfMoreOccurences
