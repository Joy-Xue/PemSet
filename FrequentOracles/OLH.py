import numpy as np
import random
import math
import xxhash
import matplotlib.pyplot as plt
import heapq
import struct
import collections


def OLH_Perturb(X, N, epsilon, c):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    # Y = np.zeros(N)
    # Y = [0] * N
    Y = [[0 for col in range(c)] for row in range(N)]
    for i in range(N):
        for j in range(c):
            Y[i][j] = xxhash.xxh32(str(X[i][j]), seed=i).intdigest() % g
            t = np.random.random()
            if t > p:
                temp = np.random.randint(g - 1)
                if temp == Y[i][j]:
                    temp = g - 1
                Y[i][j] = temp
    # print(Y)
    return Y


# k代表数据取值范围，这里为D，就不需要k了
def OLH_Aggregate(Y, N, c, epsilon, D):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g
    # print(k)
    len_D = len(D)
    # Z = np.zeros(len_D)
    # Z = [0] * len_D
    Z = [0 for col in range(len_D)]
    for i in range(len_D):
        t_count = 0
        for j in range(N):
            temp = xxhash.xxh32(str(D[i]), seed=j).intdigest() % g
            for t_j in range(c):
                # temp = xxhash.xxh32(str(D[i]), seed=j).intdigest() % g
                # if i == 0 and j < 10:
                #     print(temp)
                if temp == Y[j][t_j]:
                    t_count += 1
        Z[i] = (t_count / N - c * q) / (p - q)
    return Z


def OLH_Aggregate_file(Y, N, c, epsilon, r):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g
    file_d_name = '../../temp/olh/D' + str(r) + '.txt'
    file_dist_name = '../../temp/olh/dist' + str(r) + '.txt'
    file_d = open(file_d_name, 'r')
    file_dist = open(file_dist_name, 'w')
    str_data = file_d.readline().replace('\n', '')
    # print(str_data, xxhash.xxh32(str_data, seed=0).intdigest() % g)
    while str_data:
        t_count = 0
        # data = float(str_data)
        # print(data, type(data))
        for i in range(N):
            temp = xxhash.xxh32(str_data, seed=i).intdigest() % g
            for j in range(c):
                # temp = xxhash.xxh32(str_data, seed=i).intdigest() % g
                # print(temp)
                if temp == Y[i][j]:
                    t_count += 1
        t_count = (t_count / N - c * q) / (p - q)
        str_data = file_d.readline().replace('\n', '')
        file_dist.write(str(t_count))
        file_dist.write('\n')
    file_d.close()
    file_dist.close()
    return 0


def OLH(X, N, c, epsilon, D):
    Y = OLH_Perturb(X, N, epsilon / c, c)
    EstimateDist_OlH = OLH_Aggregate(Y, N, c, epsilon / c, D)
    # print(EstimateDist_OlH)
    return EstimateDist_OlH


# 直接将估计出来的频率存入txt文件中
def OLH_file(X, N, c, epsilon, r):
    Y = OLH_Perturb(X, N, epsilon / c, c)
    OLH_Aggregate_file(Y, N, c, epsilon / c, r)
    # print(EstimateDist_OlH)
    return 0
