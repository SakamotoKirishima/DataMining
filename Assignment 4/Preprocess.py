import pandas as pd
import numpy as np


def preprocess():
    data = pd.read_csv("nursery.data", delimiter=',')

    data_arr = np.array(data)

    parents = {'great_pret': 0, 'pretentious': 1, 'usual': 2}
    has_nurs = {'very_crit': 0, 'critical': 1, 'improper': 2, 'less_proper': 3, 'proper': 4}
    form = {'foster': 0, 'incomplete': 1, 'complete': 2, 'completed': 3}
    children = {'1': 1, '2': 2, '3': 3, 'more': 4}
    housing = {'critical': 0, 'less_conv': 1, 'convenient': 2}
    finance = {'inconv': 0,'convenient': 1}
    social = {'problematic': 0, 'slightly_prob': 1,'nonprob': 2 }
    health = {'not_recom': 0, 'priority': 1, 'recommended': 2}

    for i in range(0, len(data_arr)):
        data_arr[i, 0] = parents[data_arr[i, 0]]
        data_arr[i, 1] = has_nurs[data_arr[i, 1]]
        data_arr[i, 2] = form[data_arr[i, 2]]
        data_arr[i, 3] = children[data_arr[i, 3]]
        data_arr[i, 4] = housing[data_arr[i, 4]]
        data_arr[i, 5] = finance[data_arr[i, 5]]
        data_arr[i, 6] = social[data_arr[i, 6]]
        data_arr[i, 7] = health[data_arr[i, 7]]
    return data_arr
