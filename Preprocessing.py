import csv
from pprint import pprint

def import_data():
    attr_names=['buying', 'maint','doors','persons','lug_boot','safety', 'class']
    datafile = open('cars.csv', 'r')
    datareader = csv.reader(datafile, delimiter=',')
    data = []
    for row in datareader:
        temp_list=list()
        i=0
        for element in row:
            temp= element
            if element.isdigit():
                temp= int(element)
            temp_list.append((attr_names[i],temp))
            i=(i+1)%len(attr_names)
        data.append(temp_list)
    return data


if __name__ == '__main__':
    data = import_data()
    pprint(data)
