# -*- coding: utf-8 -*-
# !@time: 2022/10/23 16:12
# !@author: superMC @email: 18758266469@163.com
# !@fileName: base.py
import math
import random


class Base(object):
    def __init__(self):
        self.trainset = {}
        self.testset = {}
        self.n_sim_movie = 20
        self.n_rec_movie = 100
        self.max_length = 10000

    def loadfile(self, filepath):
        ''' return a generator by "yield" ,which help to save RAM. '''
        with open(filepath, 'r') as fp:
            for i, line in enumerate(fp):
                if i == self.max_length:
                    break
                yield line.strip()
        print('Load successed!')

    def data_process(self, filepath, p):
        '''
        :param filepath: rating data path
        :return: split dataset to train set and test set

        Dataset format:
        {user1:{movie1:v1, movie2:v2, ..., movieN:vN}
         user2:{...}
         ...
        }
        '''
        len_trainset = 0
        len_testset = 0
        for line in self.loadfile(filepath):
            user, movie, rating, _time = line.split('::')
            if random.random() < p:
                self.trainset.setdefault(user, {})
                # eg: 1196 {'1258': 3, '1': 4}
                self.trainset[user][movie] = int(rating)
                len_trainset += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                len_testset += 1
        print('train set len =', len_trainset)
        print('test set len =', len_testset)
        print('Trainset user count =', len(self.trainset))
        print('Testset user count =', len(self.testset))

    def recommend(self, user):
        pass

    def prediction(self, user):
        pass

    def evalute_prediction(self):
        return 0, 0

    def evalute_recommend(self, N=None):
        ''' Test recommend by precision, recall, coverage, popularity. '''
        N = N if N else self.n_rec_movie
        hit = 0
        rec_count = 0.0
        test_count = 0.0
        for i, user in enumerate(self.trainset):
            if i % 50 == 0:
                print('Recommended for %d users' % i)
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)[:N]
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
            rec_count += N
            test_count += len(test_movies)
        precision = hit / rec_count
        recall = hit / test_count
        F1 = precision * recall / (precision + recall) * 2
        return recall, F1

    def evalute(self):
        MAE, RMSE = self.evalute_prediction()
        recall, F1 = self.evalute_recommend()
        print(f"MAE:{MAE}, RMSE:{RMSE}, recall:{recall}, F1:{F1}")
        return MAE, RMSE, recall, F1
