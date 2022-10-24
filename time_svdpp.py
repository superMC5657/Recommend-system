import copy

import numpy as np
import math
import csv

from load_movie_data import loadMovieData


class timeSVDpp:

    def __init__(self, nFactors, nUsers, nItems, userItems, nBins, nDays, min_time_in_seconds, test_data, movie_set,
                 user_set):
        self.movie_set = movie_set
        self.user_set = user_set
        self.n_rec_movie = 100
        self.gamma_1 = 0.005
        self.gamma_2 = 0.007
        self.gamma_3 = 0.001
        self.g_alpha = 0.00001
        self.tau_6 = 0.005
        self.tau_7 = 0.015
        self.tau_8 = 0.015
        self.l_alpha = 0.0004

        self.max_time = nDays
        # self.min_time = 1896
        self.min_time = 0
        self.min_time_in_seconds = min_time_in_seconds

        self.userItems = userItems

        self.factors = nFactors + 1
        self.nUsers = nUsers
        self.nItems = nItems
        self.nBins = nBins
        # self.nDays = 214
        # for #users = 30, 1963 days
        self.nDays = nDays

        # initialization
        print('initialization started...')
        b_u, b_i, u_f, i_f, y_j, sumMW, bi_bin, alpha_u, bu_t, alpha_u_k, userFactors_t = self.init(self.nUsers,
                                                                                                    self.nItems,
                                                                                                    self.factors,
                                                                                                    self.nBins)

        self.bu = b_u
        self.bi = b_i
        self.bi_bin = bi_bin
        self.alpha_u = alpha_u
        self.bu_t = bu_t

        self.userFactors = u_f
        self.itemFactors = i_f
        self.y_j = y_j
        self.sumMW = sumMW

        self.alpha_u_k = alpha_u_k
        self.userFactors_t = userFactors_t
        print('initialization finished...')

        self.average = self.avg()
        print('avg = ', self.average)
        print('training started...')
        self.test_data = test_data

    def init(self, nUsers, nItems, nFactors, nBins):
        # biases
        bu = np.zeros(nUsers + 1)
        bi = np.zeros(nItems + 1, dtype='float64')

        bi_bin = np.zeros((nItems + 1, nBins))
        alpha_u = np.zeros(nUsers + 1, dtype='float64')

        bu_t = np.zeros((nUsers + 1, self.nDays))

        # factors
        userFactors = np.random.random((nUsers + 1, nFactors))
        itemFactors = np.random.random((nItems + 1, nFactors))

        y_j = np.zeros((nItems + 1, nFactors), np.float64)

        sumMW = np.random.random((nUsers + 1, nFactors))

        # time-based parameters
        alpha_u_k = np.zeros((nUsers + 1, nFactors))
        userFactors_t = []
        for j in range(nUsers + 1):
            if j % 100 == 0:
                print("u: ", j)
            f_d = np.random.random((nFactors, self.nDays))
            for i in range(len(f_d)):
                for k in range(len(f_d[i])):
                    f_d[i][k] = f_d[i][k] / math.sqrt(20)

            userFactors_t.append(f_d)

        for i in range(len(userFactors)):
            for j in range(len(userFactors[i])):
                userFactors[i][j] = userFactors[i][j] / math.sqrt(20)

        for i in range(len(itemFactors)):
            for j in range(len(itemFactors[i])):
                itemFactors[i][j] = itemFactors[i][j] / math.sqrt(20)

        for i in range(len(sumMW)):
            for j in range(len(sumMW[i])):
                sumMW[i][j] = sumMW[i][j] / math.sqrt(20)

        return bu, bi, userFactors, itemFactors, y_j, sumMW, bi_bin, alpha_u, bu_t, alpha_u_k, userFactors_t

    def train(self, iter):
        for i in range(iter):
            print('-------------------', i + 1, ' ----------------')
            self.oneIteration()
            mae, rmse = self.evalute_prediction()
            print('iteration: ', i + 1, ', MAE = ', mae, ', RMSE = ', rmse)

    def oneIteration(self):

        for userId in range(1, len(self.userItems) + 1):

            print('updates for the user:', userId, ' STARTED...')

            tmpSum = np.zeros(self.factors, dtype='float')

            sz = len(self.userItems[userId])

            if sz > 0:

                sqrtNum = 1 / (math.sqrt(sz))

                for f in range(self.factors):
                    sum_y = 0
                    for j in range(sz):
                        pos_item = self.userItems[userId][j][0]
                        sum_y += self.y_j[pos_item][f]
                    self.sumMW[userId][f] = sum_y

                for it in range(sz):
                    itemid = self.userItems[userId][it][0]
                    rating = self.userItems[userId][it][1]
                    timestamp_ = self.userItems[userId][it][2]

                    prediction = self.prediction(userId, itemid, timestamp_)
                    error = rating - prediction

                    self.bu[userId] += 0.01 * error - 0.01 * self.bu[
                        userId]  # self.gamma_1 * error - self.tau_6 * self.bu[userId]
                    self.bi[itemid] += 0.01 * error - 0.01 * self.bi[
                        itemid]  # self.gamma_1 * error - self.tau_6 * self.bi[itemid]

                    self.bu_t[userId][timestamp_] += 0.01 * (error - 0.01 * self.bu_t[userId][
                        timestamp_])  # self.gamma_1 * (error - 0.005 * self.bu_t[userId - 1][timestamp_])
                    self.bi_bin[itemid][self.calBin(timestamp_)] += 0.01 * (error - 0.01 * self.bi_bin[itemid][
                        self.calBin(
                            timestamp_)])  # self.gamma_1 * (error - 0.005 * self.bi_bin[itemid][self.calBin(timestamp_)])
                    self.alpha_u[userId] += 0.01 * (error * self.dev(userId, timestamp_) - 0.01 * self.alpha_u[
                        userId])  # self.g_alpha * (error * self.dev(userId, timestamp_) - self.l_alpha * self.alpha_u[userId])

                    # updating factors
                    for k in range(self.factors):
                        u_f = self.userFactors[userId][k]
                        i_f = self.itemFactors[itemid][k]
                        u_f_t = self.userFactors_t[userId][k][timestamp_]

                        self.userFactors[userId][k] += 0.01 * (
                                error * i_f - 0.01 * u_f)  # self.gamma_1 * (error * i_f - 0.015 * u_f)
                        self.itemFactors[itemid][k] += 0.01 * (error * (u_f + sqrtNum * self.sumMW[userId][
                            k]) - 0.01 * i_f)  # self.gamma_1 * (error * (u_f + sqrtNum * self.sumMW[userId][k]) - 0.015 * i_f)
                        self.alpha_u_k[userId][k] += 0.01 * (
                                error * self.dev(userId, timestamp_) - 0.01 * self.alpha_u_k[userId][
                            k])  # self.g_alpha * (error * self.dev(userId, timestamp_) - self.l_alpha * self.alpha_u_k[userId][k])
                        self.userFactors_t[userId][k][timestamp_] += 0.01 * (
                                error * i_f - 0.01 * u_f_t)  # self.gamma_1 * (error * i_f - 0.015 * u_f_t)
                        tmpSum[k] += error * sqrtNum * i_f

                for j in range(sz):
                    itID = self.userItems[userId][j][0]
                    for f in range(self.factors):
                        tmpMW = self.y_j[itID][f]
                        self.y_j[itID][f] += 0.01 * (
                                tmpSum[f] - 0.01 * tmpMW)  # self.gamma_1 * (tmpSum[f] - 0.015 * tmpMW)
                        self.y_j[itID][f] = round(self.y_j[itID][f], 4)
                        self.sumMW[userId][f] += self.y_j[itID][f] - tmpMW

        for userId in range(1, len(self.userItems) + 1):
            sz = len(self.userItems[userId])
            if sz > 0:
                sqrtNum = 1 / (math.sqrt(sz))
                for k in range(self.factors):
                    sumy = 0
                    for i in range(sz):
                        itID = self.userItems[userId][i][0]
                        sumy += self.y_j[itID][k]
                    self.sumMW[userId][k] = sumy

        self.gamma_1 *= 0.9
        self.g_alpha *= 0.9

    # overall rating avarage
    def avg(self):
        s = 0
        count = 0
        # l = len(self.matrix)
        l = len(self.userItems)
        for i in range(1, l + 1):

            sz = len(self.userItems[i])
            for j in range(sz):
                s += self.userItems[i][j][1]
                count += 1
        avg = s / count

        return avg

    # find the index of the bin for the given timestamp
    def calBin(self, day_of_rating):
        interval = (self.max_time - self.min_time) / self.nBins
        bin_ind = np.minimum(self.nBins - 1, int((day_of_rating - self.min_time) / interval))

        return bin_ind

    # deviation of user u at given t
    def dev(self, userID, t):
        deviation = np.sign(t - self.meanTime(userID)) * pow(abs(t - self.meanTime(userID)), 0.015)

        return deviation

    # mean rating time for given user
    def meanTime(self, userID):
        s = 0
        count = 0
        sz = len(self.userItems[userID])
        if sz > 0:
            for i in range(sz):
                s += self.userItems[userID][i][2]
                count += 1
            return s / count
        else:
            return 0

    # prediction method
    def prediction(self, u, i, day_ind):
        sz = len(self.userItems[u])
        if sz > 0:
            sqrtNum = 1 / math.sqrt(sz)
        else:
            sqrtNum = 0

        tmp = 0
        for k in range(self.factors):
            tmp += ((self.userFactors[u][k] + self.alpha_u_k[u][k] * self.dev(u, day_ind) + self.userFactors_t[u][k][
                day_ind]) + (sqrtNum * self.sumMW[u][k])) * self.itemFactors[i][k]
        prediction = self.average + self.bu[u] + self.bi[i] + self.bi_bin[i][self.calBin(day_ind)] + self.alpha_u[
            u] * self.dev(u, day_ind) + self.bu_t[u][day_ind] + tmp

        return prediction

    # evaluating the model using RMSE
    def evalute_prediction(self):

        MSE = 0
        c = 0
        MAE = 0.0

        for row in self.test_data:
            userid = int(row[0])
            itemid = int(row[1])
            rating = float(row[2])
            tmp = int(row[3])

            day = min(self.nDays - 1, int((tmp - self.min_time_in_seconds) / 86400))

            predict = self.prediction(userid, itemid, day)
            MSE += math.pow((rating - predict), 2)
            MAE += abs((rating - predict))

            c += 1

        MSE = MSE / c
        MAE = MAE / c
        RMSE = math.sqrt(MSE)

        return MAE, RMSE

    def recommend(self, user, time):
        ''' 产生预测评分最高的N 个推荐结果，默认返回N个

         :return 返回前N 个评分值的item，及其预测评分值. 格式[(item1,score1), (item2,score2), ...]
         '''
        unratedItems = copy.deepcopy(self.movie_set)
        for info in self.userItems[user]:
            item_id = info[0]
            unratedItems.remove(item_id)
        if len(unratedItems) == 0:
            return 'you rated everything'  # 如果都已经评过分，则退出
        itemScores = []
        for item in list(unratedItems):  # 对于每个未评分的item，都计算其预测评分
            r_hat = self.prediction(user, item, time)
            itemScores.append((item, r_hat))
        itemScores = sorted(
            itemScores,
            key=lambda x: x[1],
            reverse=True)  # 按照item的得分进行从大到小排序
        return itemScores[:self.n_rec_movie]

    def evalute_recommend(self, N=None):
        ''' Test recommend by precision, recall, coverage, popularity. '''
        N = N if N else self.n_rec_movie
        hit = 0
        rec_count = 0.0
        test_count = 0.0
        for i, user in enumerate(self.user_set):
            if i % 50 == 0:
                print('Recommended for %d users' % i)
            test_movies = []
            for info in self.userItems[user]:
                test_movies.append(info[0])
            last_time = self.userItems[user][-1][2]
            rec_movies = self.recommend(user, last_time)[:N]
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
            rec_count += N
            test_count += len(test_movies)
        precision = hit / rec_count
        recall = hit / test_count
        F1 = precision * recall / (precision + recall) * 2
        return recall, F1


# M A I N

lm = loadMovieData()

userItems, nUsers, nItems, nDays, minTimestamp, test_data, movie_set, user_set = lm.main("Data_raw/ml-1m/ratings.dat")
nFactors = 10
nBins = 6

timesvd_pp = timeSVDpp(nFactors, nUsers, nItems, userItems, nBins, nDays, minTimestamp, test_data, movie_set, user_set)
timesvd_pp.train(1)
timesvd_pp.evalute_recommend(10)

# 10, bins, 1 iteration: RMSE = 1.02648993887
# 10 bins,  20 iterations: RMSE = 0.94744760782
# 10 bins, 30 iterations: RMSE = 0.94331981365

# 30 bins, 20 iterations: 0.94672648969
