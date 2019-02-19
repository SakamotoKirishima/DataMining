import csv
from pprint import pprint


class Import:
    """
    Class to import dataset and put it into a usable format
    """
    @staticmethod
    def import_data():
        """
        Static method to import data
        :return: Dictionary with each row of the dataset as the key and 1 as the value
        """
        attr_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        datafile = open('cars.csv', 'r')
        datareader = csv.reader(datafile, delimiter=',')
        data = []
        for row in datareader:
            temp_list = list()
            i = 0
            for element in row:
                temp = element
                if element.isdigit():
                    temp = int(element)
                temp_list.append((attr_names[i], temp))
                i = (i + 1) % len(attr_names)
            data.append(temp_list)
        retDict = dict()
        for row in data:
            temp_tuple= list()
            for element in row:
                if element not in temp_tuple:
                    temp_tuple.append(element)

            retDict[tuple(temp_tuple)] = 1
        return retDict


if __name__ == '__main__':
    data = Import.import_data()
    pprint(data)
