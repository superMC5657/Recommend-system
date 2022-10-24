# -*- coding: utf-8 -*-
# !@time: 2022/10/23 16:12
# !@author: superMC @email: 18758266469@163.com
# !@fileName: item_cf.py


import time
import math
import random
from collections import defaultdict
from operator import itemgetter

from base import Base


class ItemBasedCF(Base):
    def __init__(self):
        super(ItemBasedCF, self).__init__()
        self.movie_simmat = {}
        self.movie_popular = {}
        self.movie_count = 0.0
        self.max_length = 10000

    def calculate_movie_sim(self):
        count = 0
        for user, movies in self.trainset.items():  # items 返回键值对
            for m in movies:
                self.movie_popular[m] = self.movie_popular.get(
                    m, 0) + 1  # 电影m 每次被不同人评价过 +1
                self.movie_simmat.setdefault(m, defaultdict(int))
                for m2 in movies:
                    if m != m2:
                        # 如果两部电影被同一个人评价过，那么sim +1
                        self.movie_simmat[m][m2] += 1

            if count % 1000 == 0:
                print('calu movie sim ... (%d)' % count)
            count += 1
            # print(user,movies)
        self.movie_count = len(self.movie_popular)
        cal_sim_count = 0
        for m1, related_movies in self.movie_simmat.items():
            for m2, count in related_movies.items():
                self.movie_simmat[m1][m2] = count / \
                                            math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
                cal_sim_count += 1
                if cal_sim_count % 2000000 == 0:
                    print(
                        'calculating item similarity ... (%d)' %
                        cal_sim_count)

    def recommend(self, user):
        ''' Find K similar movies and recommend N movies based on user watched. '''
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainset[user]
        for movie, rating in watched_movies.items():
            for related_movie, similarity in sorted(
                    self.movie_simmat[movie].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                rank[related_movie] = rank.get(
                    related_movie, 0) + similarity % rating
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def prediction(self, user):
        ''' Predict all movie for user based on user watched

         :return rank: { m1:rating1, m2:rating2, ...} 对于用户user所有未打分的电影的预测分值
         '''
        rank = {}
        watched_movies = self.trainset[user]
        for movie, rating in watched_movies.items():
            for other_movie, similarity in self.movie_simmat[movie].items():
                if other_movie in watched_movies:
                    continue
                rank[other_movie] = rank.get(other_movie, 0) + similarity * rating
        len_ = len(watched_movies)
        if len_ != 0:
            for m, rating in rank.items():
                # print(m,rating,type(m),type(rating),len_)
                rank[m] = 5 * rating / len_  # 乘以5 是因为评分是0-5的，不然算出来是个0-1 左右的值
        return rank


    def evalute_prediction(self):
        ''' Test prediction by MSE. '''
        MSE = 0.0
        MAE = 0.0
        eval_count = 0
        for i, user in enumerate(self.testset):
            if i % 500 == 0:
                print('Prediction for %d users in testset.' % i)
            test_movie_score = self.testset.get(user, {})
            rec_movie_score = self.prediction(user)
            for m, real_score in test_movie_score.items():
                temp = rec_movie_score.get(m, 0) - real_score
                eval_count += 1
                MSE += temp ** 2
                MAE += abs(temp)
                if eval_count % 1000 == 0:
                    print('eval_count(%d) user:%s to movie %s, real_score:%f, predict_score:%f, error:%f' % (
                        eval_count, user, m, real_score, rec_movie_score.get(m, 0), temp))
        MSE /= eval_count
        MAE /= eval_count
        RMSE = math.sqrt(MSE)
        return MAE, RMSE


def main():
    print('*' * 20, 'Item-based collaborative filtering algorithm', '*' * 20)
    itemcf = ItemBasedCF()
    itemcf.data_process('./Data_raw/ml-1m/ratings.dat', p=0.8)
    # time_s = time.time()
    itemcf.calculate_movie_sim()
    # time_m = time.time()
    # itemcf.evalute_recommend()
    # time_er = time.time()
    # itemcf.evalute_prediction()
    # time_ep = time.time()
    # print('Time spent calculating is:', time_m - time_s)
    # # print('Time spent on recommendations:',time_er - time_m)
    # print('Time spend predicting is:', time_ep - time_er)
    itemcf.evalute()


if __name__ == '__main__':
    main()
