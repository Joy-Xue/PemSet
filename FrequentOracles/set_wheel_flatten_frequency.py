import numpy as np
import random
import math
import xxhash
import matplotlib.pyplot as plt
import heapq
import struct
import collections
import os


# X是用户原始数据，d是可能取到的最大值加1，flat_num是将重复次数大于它的数据近似地看成只重复了这么多次。
# 不能直接在原始数据上操作，而是应该创建一个新列表存数据。
def flat_data(X, N, c, d, flat_num):
    X1 = []
    for i in range(N):
        X1.append([])
        temp_data = collections.Counter(X[i])
        if max(temp_data.values()) > 1:
            data_1 = filter(lambda x: x[1] == 1, temp_data.items())
            for (t1_1, t1_2) in data_1:
                X1[i].append(t1_1)
            for j in range(2, max(temp_data.values()) + 1):
                data_j = filter(lambda x: x[1] == j, temp_data.items())
                for (tj_1, tj_2) in data_j:
                    if j < flat_num:
                        for t_j in range(j):
                            t_index = X[i].index(tj_1)
                            X1[i].append(X[i][t_index] + (j - t_j - 1) * d)
                            # X[i][t_index] = X[i][t_index] + (j - t_j - 1) * d
                    else:
                        for t_j in range(j):
                            t_index = X[i].index(tj_1)
                            if j - t_j >= flat_num:
                                X1[i].append(X[i][t_index] + (flat_num - 1) * d)
                                # X[i][t_index] = X[i][t_index] + (flat_num - 1) * d
                            else:
                                X1[i].append(X[i][t_index] + (j - t_j - 1) * d)
                                # X[i][t_index] = X[i][t_index] + (j - t_j - 1) * d
        else:
            for j in range(c):
                X1[i].append(X[i][j])
        # print(X[i], X1[i])
        # if len(X1[i]) == 0:
        #     os.system("pause")
    return X1


# 扰动过程, X为用户的真实数据, epsilon
def wheel_perturb_set_data_flatten_frequency(X, N, c, epsilon):
    max_int_32 = (1 << 32) - 1
    # Y = [0] * N
    Y = [0 for col in range(N)]
    # Y = np.zeros(N, dtype=float)
    s = math.exp(epsilon)
    temp_p = 1 / (2 * c - 1 + c * s)
    omega = c * temp_p * s + (1 - c * temp_p)
    for i in range(N):
        # V = [0] * c
        V = [0 for col in range(c)]
        # V = np.zeros(c, dtype=float)
        # hash
        # print(X[i])
        for j in range(c):
            # print(j)
            V[j] = xxhash.xxh32_intdigest(str(X[i][j]), seed=i) / max_int_32
        # 区间合并的准备工作
        bSize = math.ceil(1 / temp_p)
        # lef = [0] * bSize
        # rig = [0] * bSize
        lef = [0 for col in range(bSize)]
        rig = [0 for col in range(bSize)]
        # lef = np.zeros(bSize, dtype=float)
        # rig = np.zeros(bSize, dtype=float)
        for b in range(bSize):
            lef[b] = min((b + 1) * temp_p, 1.0)
            rig[b] = b * temp_p
        for v in V:
            temp_b = math.ceil(v / temp_p) - 1
            lef[temp_b] = min(v, lef[temp_b])
            if temp_b < math.ceil(1 / temp_p) - 1:
                rig[temp_b + 1] = max(v + temp_p, rig[temp_b + 1])
            else:
                rig[0] = max(v + temp_p - 1, rig[0])
        temp_rig0 = rig[0]
        for b in range(bSize - 1):
            lef[b] = max(lef[b], rig[b])
            rig[b] = rig[b + 1]
        lef[bSize - 1] = max(lef[bSize - 1], rig[bSize - 1])
        rig[bSize - 1] = temp_rig0 + 1.0
        ll = 0.0
        for b in range(bSize):
            ll = ll + rig[b] - lef[b]
        r = np.random.random_sample()
        a = 0.0
        # 师兄写错了应该是bSize
        # for b in range(bSize - 1):
        for b in range(bSize):
            a = a + s * (rig[b] - lef[b]) / omega
            if a > r:
                z = rig[b] - (a - r) * omega / s
                break
            a = a + (omega - ll * s) * (lef[(b + 1) % round(bSize)] + math.floor((b + 1) * temp_p) - rig[b]) / (
                    (1 - ll) * omega)
            if a > r:
                z = lef[(b + 1) % bSize] - (a - r) * (1 - ll) * omega / (omega - ll * s)
                break
        z = z % 1.0
        Y[i] = z
    return Y


# 用户输入集合值聚合过程
def wheel_aggregate_set_data_flatten_frequency(Y, N, c, epsilon, D, d, flat_num):
    max_int_32 = (1 << 32) - 1
    k = len(D)
    # Estimate_Dist = [0] * k
    temp_k = k * flat_num
    Estimate_Dist_flat = [0 for col in range(temp_k)]
    Estimate_Dist = [0 for col in range(k)]
    # Estimate_Dist = np.zeros(k, dtype=float)
    s = math.exp(epsilon)
    temp_p = 1 / (2 * c - 1 + c * s)
    for i in range(N):
        z = Y[i]
        for j in range(k):
            for t_flat in range(flat_num):
                x = D[j] + t_flat * d
                v = xxhash.xxh32_intdigest(str(x), seed=i) / max_int_32
                if z - temp_p < v <= z or z - temp_p + 1 < v < 1:
                    Estimate_Dist_flat[j + t_flat * k] += 1
            # for t_flat in range(flat_num):
            #     x = D[j] + t_flat * d
            #     v = xxhash.xxh32_intdigest(str(x), seed=i) / max_int_32
            #     if z - temp_p < v <= z or z - temp_p + 1 < v < 1:
            #         Estimate_Dist[j] += 1
    # 矫正过程
    pt = temp_p * s / (c * temp_p * s + (1 - c * temp_p))
    pf = temp_p
    for i in range(k):
        for j in range(flat_num):
            Estimate_Dist[i] += 1 / N * (Estimate_Dist_flat[i + j * k] - N * pf) / (pt - pf)
    return Estimate_Dist


def wheel_aggregate_set_data_file_flatten_frequency(Y, N, c, epsilon, r, d, flat_num):
    max_int_32 = (1 << 32) - 1
    s = math.exp(epsilon)
    temp_p = 1 / (2 * c - 1 + c * s)
    pt = temp_p * s / (c * temp_p * s + (1 - c * temp_p))
    pf = temp_p
    file_d_name = '../../temp/wheel_flatten_frequency/D' + str(r) + '.txt'
    file_dist_name = '../../temp/wheel_flatten_frequency/dist' + str(r) + '.txt'
    file_d = open(file_d_name, 'r')
    file_dist = open(file_dist_name, 'w')
    str_data = file_d.readline().replace('\n', '')
    # print(str_data, xxhash.xxh32(str_data, seed=0).intdigest() % g)
    while str_data:
        t_frequency = 0
        # t_count = 0
        # data = float(str_data)
        # print(data, type(data))
        for t_flat in range(flat_num):
            t_count = 0
            x_data = int(str_data) + t_flat * d
            for i in range(N):
                z = Y[i]
                v = xxhash.xxh32_intdigest(str(x_data), seed=i) / max_int_32
                if z - temp_p < v <= z or z - temp_p + 1 < v < 1:
                    t_count += 1
            t_frequency += 1 / N * (t_count - N * pf) / (pt - pf)
        str_data = file_d.readline().replace('\n', '')
        file_dist.write(str(t_frequency))
        file_dist.write('\n')
    file_d.close()
    file_dist.close()
    return 0


def set_wheel_flatten_frequency(X, N, c, epsilon, D, d, flat_num):
    X1 = flat_data(X, N, c, d, flat_num)
    Y = wheel_perturb_set_data_flatten_frequency(X1, N, c, epsilon)
    EstimateDist_wheel = wheel_aggregate_set_data_flatten_frequency(Y, N, c, epsilon, D, d, flat_num)
    return EstimateDist_wheel


# 直接将估计出来的频率存入txt文件中
def set_wheel_file_flatten_frequency(X, N, c, epsilon, r, d, flat_num):
    X1 = flat_data(X, N, c, d, flat_num)
    Y = wheel_perturb_set_data_flatten_frequency(X1, N, c, epsilon)
    wheel_aggregate_set_data_file_flatten_frequency(Y, N, c, epsilon, r, d, flat_num)
    return 0
