# -*- coding: utf-8 -*-
# !@time: 2022/10/23 20:49
# !@author: superMC @email: 18758266469@163.com
# !@fileName: load_movie_data.py


__author__ = 'trimi'

import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import itemgetter
import struct
import gzip
import numpy as np
import matplotlib as mp
import random
from random import randint
import math
import matplotlib.pyplot as plt
import collections


class loadMovieData:
    def __init__(self, max_length=10000, p=0.8):
        self.max_length = max_length
        self.p = p

    def read_training_data(self, path):
        movie_set = set()
        user_set = set()
        with open(path, 'r') as f:
            matrix = []
            userItems = {}
            itemUsers = {}
            max_item = []
            timestamps = []
            b = 0
            test_data = []

            for index, line in enumerate(f):

                row = []
                if index == self.max_length:
                    break

                a = line.strip().split('::')

                user = int(a[0])
                item = int(a[1])
                rating = float(a[2])
                time_ = int(a[3])
                movie_set.add(item)
                user_set.add(user)
                if np.random.random() < 1 - self.p:
                    test_data.append((user, item, rating, time_))
                    continue
                row.append(user)
                row.append(item)
                row.append(rating)
                row.append(time_)

                matrix.append(row)
                max_item.append(item)
                timestamps.append(time_)

                # pos events per user
                if user not in userItems:
                    userItems[user] = [(item, rating, time_)]
                else:
                    if item not in userItems[user]:
                        userItems[user].append((item, rating, time_))

                # items rated by users
                if item not in itemUsers:
                    itemUsers[item] = [(user, rating, time_)]
                else:
                    if user not in itemUsers[item]:
                        itemUsers[item].append((user, rating, time_))

                b += 1
            print('#pos_events = ', b)
            min_timestamp = min(timestamps)
            max_timestamp = max(timestamps)

            print('max item id = ', max(max_item))
            return matrix, userItems, itemUsers, min_timestamp, max_timestamp, test_data, movie_set, user_set

    def num_of_days(self, min_timestamp, max_timestamp):
        nDays = int((max_timestamp - min_timestamp) / 86400)

        return nDays

    def cal_day(self, timestamp, num_of_days, min_timestamp):
        day_ind = np.minimum(num_of_days - 1, int((timestamp - min_timestamp) / 86400))

        return day_ind

    def timestamp_to_day(self, matrix, n_days, min_timestamp):

        userItems = {}
        itemUsers = {}

        for i in range(len(matrix)):
            user = matrix[i][0]
            item = matrix[i][1]
            rating = matrix[i][2]
            timestamp = matrix[i][3]

            day_ind = self.cal_day(timestamp, n_days, min_timestamp)

            # pos events per user
            if user not in userItems:
                userItems[user] = [(item, rating, day_ind)]
            else:
                if item not in userItems[user]:
                    userItems[user].append((item, rating, day_ind))

            # items rated by users
            if item not in itemUsers:
                itemUsers[item] = [(user, rating, day_ind)]
            else:
                if user not in itemUsers[item]:
                    itemUsers[item].append((user, rating, day_ind))

        return userItems, itemUsers

    def main(self, filepath):
        matrix, userItems, itemUsers, min_timestamp, max_timestamp, test_data, movie_set, user_set = self.read_training_data(
            filepath)
        num_days = self.num_of_days(min_timestamp, max_timestamp)
        new_userItems, new_itemUsers = self.timestamp_to_day(matrix, num_days, min_timestamp)

        nUsers = len(new_userItems)
        nItems = 3952

        return new_userItems, nUsers, nItems, num_days, min_timestamp, test_data, movie_set, user_set
