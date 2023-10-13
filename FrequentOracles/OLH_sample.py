import numpy as np
import random
import math
import xxhash
import matplotlib.pyplot as plt
import heapq
import struct
import collections


def set_to_single(X, N, c):
    # S_X = np.zeros(N)
    # S_X = [0] * N
    S_X = [0 for col in range(N)]
    for i in range(N):
        S_X[i] = X[i][np.random.randint(c)]
        # print(X[i], S_X[i])
    return S_X


def OLH_Perturb_sample(S_X, N, epsilon):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    # Y = np.zeros(N)
    # Y = [0] * N
    Y = [0 for col in range(N)]
    for i in range(N):
        Y[i] = xxhash.xxh32(str(S_X[i]), seed=i).intdigest() % g
        t = np.random.random()
        if t > p:
            temp = np.random.randint(g - 1)
            if temp == Y[i]:
                temp = g - 1
            Y[i] = temp
    # print(Y)
    return Y


# k代表数据取值范围，这里为D，就不需要k了
def OLH_Aggregate_sample(Y, N, c, epsilon, D):
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
            # if i == 0 and j < 10:
            #     print(temp)
            if temp == Y[j]:
                t_count += 1
        Z[i] = c * (t_count / N - q) / (p - q)
    return Z


def OLH_Aggregate_sample_file(Y, N, c, epsilon, r):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g
    file_d_name = '../../temp/olh_sample/D' + str(r) + '.txt'
    file_dist_name = '../../temp/olh_sample/dist' + str(r) + '.txt'
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
            # print(temp)
            if temp == Y[i]:
                t_count += 1
        t_count = c * (t_count / N - q) / (p - q)
        str_data = file_d.readline().replace('\n', '')
        file_dist.write(str(t_count))
        file_dist.write('\n')
    file_d.close()
    file_dist.close()
    return 0


def OLH_sample(X, N, c, epsilon, D):
    S_X = set_to_single(X, N, c)
    Y = OLH_Perturb_sample(S_X, N, epsilon)
    EstimateDist_OlH = OLH_Aggregate_sample(Y, N, c, epsilon, D)
    # print(EstimateDist_OlH)
    return EstimateDist_OlH


# 直接将估计出来的频率存入txt文件中
def OLH_sample_file(X, N, c, epsilon, r):
    S_X = set_to_single(X, N, c)
    Y = OLH_Perturb_sample(S_X, N, epsilon)
    OLH_Aggregate_sample_file(Y, N, c, epsilon, r)
    # print(EstimateDist_OlH)
    return 0
