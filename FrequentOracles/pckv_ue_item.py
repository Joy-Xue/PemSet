import numpy as np
import random
import math
import xxhash
import matplotlib.pyplot as plt
import heapq
import struct
import collections
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix

# 主要问题是用户扰动后的数据量也非常大，没办法直接存下来。
# 用lil_matrix存储数据，并且将1和-1分开存储，这样就可以直接使用l.sum(axis=0)来统计列上一共有多少个1或-1了。
# 计数时稀疏矩阵也没法处理2^64这么大的，会报错，转成numpy处理也会有错。
# 不把perturb和aggregate的过程分开了，放在一起写。
# 但是也需要对每位上1和-1的个数存储，这时不管是用numpy或者是list也都不行啊。
# 使用文件存？或者分段处理？
# dok矩阵最大只能这么大，lil更小。
# n = pow(2, 62)
# t = pow(2, 62)
# p = dok_matrix((n, t))
# p[0, 0] = 2
# p[pow(2, 62)-1, 5] = 3
# print(p, p[0], p[pow(2, 62)-1])
# 对于所有需要估计的数据，直接在用户数据上处理得到结果
# global max_v, min_v
# max_v = 0
# # 其实min_v应该设置为c
# min_v = 100


# append的时候就不用遍历所有键可能的取值了
def set_to_kv(X, N, c):
    global max_v, min_v
    k_v = []
    for i in range(N):
        k_v.append([])
    for i in range(N):
        temp_data = collections.Counter(X[i])
        t_keys = sorted(temp_data.keys())
        temp = dict(temp_data)
        for j in range(len(t_keys)):
            t_key = t_keys[j]
            t_value = temp[t_key]
            k_v[i].append([t_key, t_value])
            # if t_value < min_v:
            #     min_v = t_value
            # elif t_value > max_v:
            #     max_v = t_value
    for i in range(N):
        for j in range(len(k_v[i])):
            k_v[i][j][1] = 2 * k_v[i][j][1] / (c - 1) + (-c - 1) / (c - 1)
            # k_v[i][j][1] = (2 * k_v[i][j][1] - min_v - max_v) / (max_v - min_v)
    # print(k_v)
    return k_v


# 补齐到c个然后抽样，d是目前取到的最大值加1
def P_S(X, l, N, d):
    k_v = [[0 for col in range(2)] for row in range(N)]
    # k_v = [[0] * 2 for _ in range(N)]
    # k_v = np.zeros((N, 2), dtype=int)
    for i in range(N):
        # flag = 0
        S = len(X[i])
        b = S / (max(S, l))
        # b = 1
        rnd = np.random.random()
        if rnd < b:  # 从用户项集S中随机选择一个kv对，从[0,S)中随机选择一个数作为序号
            # flag = 1
            tmp = random.sample(X[i], 1)
            k_ps = tmp[0][0]
            v_ps = tmp[0][1]
        else:
            v_ps = 0
            k_ps = random.randint(d, d + l - 1)
            # k_ps = np.random.randint(d, d + l)  # 从{d+1,d+2,...,d'}中随机选择一个作为序号
        # 离散v_ps值
        p = (1 + v_ps) / 2
        rnd = np.random.random()
        if rnd < p:
            v_ps = 1
        else:
            v_ps = -1
        k_v[i][0] = k_ps
        k_v[i][1] = v_ps
        # if flag == 1:
        #     print(tmp, k_v[i])
    return k_v


# 针对某一条需要估计的数据x对相应位进行扰动
def perturb_item(x, k, v, epsilon):
    epsilon1 = math.log((math.exp(epsilon) + 1) / 2)
    epsilon2 = epsilon
    a = 1 / 2
    b = 1 / (math.exp(epsilon1) + 1)
    p = math.exp(epsilon2) / (math.exp(epsilon2) + 1)
    r = np.random.random()
    if x == k:
        if r < a * p:
            t_value = v
        elif r < a:
            t_value = -v
        else:
            t_value = 0
    else:
        if r < b / 2:
            t_value = 1
        elif r < b:
            t_value = -1
        else:
            t_value = 0
    return t_value


# 针对某一条需要估计的数据进行aggregate
def aggregate_item(n1, n2, l, epsilon, N, c):
    epsilon1 = math.log((math.exp(epsilon) + 1) / 2)
    epsilon2 = epsilon
    a = 1 / 2
    b = 1 / (math.exp(epsilon1) + 1)
    p = math.exp(epsilon2) / (math.exp(epsilon2) + 1)
    t1 = a * p - b / 2
    t2 = a * (1 - p) - b / 2
    A = np.array([[t1, t2], [t2, t1]])
    A1 = np.linalg.inv(A)
    # 估计键的频率
    fk = l * ((n1 + n2) / N - b) / (a - b)
    # fk[i]如果是0那么估计mk[i]时的分母也为0，就没法估计了，实际应用中用户数很大，应该每个key都至少有一个人的
    if fk < 1 / N:
        fk = 1 / N
    elif fk > 1:
        fk = 1
    # 估计值的均值
    t = np.dot(A1, [[n1 - N * b / 2], [n2 - N * b / 2]])
    # 这里t0和t1都是[]（np.ndarray）类型
    t_n1 = float(t[0])
    if t_n1 < 0:
        t_n1 = 0
    elif t_n1 > N * fk / l:
        t_n1 = N * fk / l
    t_n2 = float(t[1])
    if t_n2 < 0:
        t_n2 = 0
    elif t_n2 > N * fk / l:
        t_n2 = N * fk / l
    mk = l * (t_n1 - t_n2) / (N * fk)
    # print(A1)
    # print(t, t_n1, t_n2, mk, type(mk))
    # 将值缩放回原来的数
    tmk = (c - 1) * mk / 2 + (c + 1) / 2
    # tmk = (mk * (max_v - min_v) + max_v + min_v) / 2
    return fk, tmk


# D是需要估计的集合，d是本轮中可能取到的最大值，这里令l等于c
def PCKV_UE(X, N, c, epsilon, D, d):
    global max_v, min_v
    len_D = len(D)
    X_t = set_to_kv(X, N, c)
    k_v = P_S(X_t, c, N, d)
    EstimateDist = [0 for col in range(len_D)]
    for i in range(len_D):
        n1 = 0
        n2 = 0
        t_key = D[i]
        for j in range(N):
            k = k_v[j][0]
            v = k_v[j][1]
            t_value = perturb_item(t_key, k, v, epsilon)
            if t_value == 1:
                n1 += 1
            elif t_value == -1:
                n2 += 1
        fk, tmk = aggregate_item(n1, n2, c, epsilon, N, c)
        EstimateDist[i] = fk * tmk
    return EstimateDist


def PCKV_UE_file(X, N, c, epsilon, r, d):
    file_d_name = '../../temp/pckv_ue/D' + str(r) + '.txt'
    file_dist_name = '../../temp/pckv_ue/dist' + str(r) + '.txt'
    file_d = open(file_d_name, 'r')
    file_dist = open(file_dist_name, 'w')
    str_data = file_d.readline().replace('\n', '')
    X_t = set_to_kv(X, N, c)
    k_v = P_S(X_t, c, N, d)
    while str_data:
        t_key = int(str_data)
        n1 = 0
        n2 = 0
        for j in range(N):
            k = k_v[j][0]
            v = k_v[j][1]
            t_value = perturb_item(t_key, k, v, epsilon)
            if t_value == 1:
                n1 += 1
            elif t_value == -1:
                n2 += 1
        fk, tmk = aggregate_item(n1, n2, c, epsilon, N, c)
        EstimateDist = fk * tmk
        str_data = file_d.readline().replace('\n', '')
        file_dist.write(str(EstimateDist))
        file_dist.write('\n')
        # print(fk, tmk, str(EstimateDist))
    file_d.close()
    file_dist.close()
    return 0
