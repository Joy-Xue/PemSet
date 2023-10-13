import numpy as np
import random
import math
import xxhash
import matplotlib.pyplot as plt
import heapq
import struct
import collections
from OLH import OLH, OLH_file
from OLH_sample import OLH_sample, OLH_sample_file
from set_wheel import set_wheel, set_wheel_file
from pckv_ue_item import PCKV_UE, PCKV_UE_file
from read_data_dist import read_data


# numpy是用C实现的，最大只能存到uint64=2^64-1，
# 整数转浮点数，当数字太大的时候又会出问题 比如float(1013431525348850692)=1013431525348850688.000000
# 因此换用list进行存储
# 将各种方法都写成函数
# 修改m需要修改主函数和59，107行
# 根据D中的数据量，判断将估计出来的频率存储在数组中还是文件中
# 百度说list最多存1152921504606846975=2^60个数，但是试了不行，这里设置list最多存2^31个数

# 需要在每个函数中再定义一次
np.set_printoptions(suppress=True)


def data_to_prefix(X, N, g, c, m, k):
    # 第一位不用从第二位开始存
    Ni = [0 for col in range(g + 1)]
    si = [0 for col in range(g + 1)]
    prefix_X = [[0 for col in range(c)] for row in range(N)]
    # Ni = [0] * (g + 1)
    r_N = math.ceil(N / g)
    # prefix_X = [[0] * c for _ in range(N)]
    for i in range(1, g + 1):
        r_start = (i - 1) * r_N
        b_start = (i - 1)
        if i == g:
            r_stop = N
        else:
            r_stop = i * r_N
        # Ni[i] = r_stop - r_start
        s = math.ceil(math.log2(k)) + math.ceil(i * (m - math.ceil(math.log2(k))) / g)
        si[i] = s
        # print(k, i, s)
        for j in range(r_start, r_stop):
            for t_j in range(c):
                # print(X[i][j], int(X[i][j]))
                # 整数转二进制并补齐到64位，截取长度为s的前缀，并转回整数
                # print(X[i][j])
                prefix_X[j][t_j] = int('{:0160b}'.format(int(X[j][t_j]))[0:s], 2)
    return prefix_X, si


def split_prefix_X(prefix_X, r, g, N):
    r_N = math.ceil(N / g)
    r_start = (r - 1) * r_N
    if r == g:
        r_stop = N
    else:
        r_stop = r * r_N
    t_X = prefix_X[r_start:r_stop]
    return t_X, r_stop - r_start


# 拼接二进制字符串得到集合D，每轮只用估计D中的数据即可
def construct_D(Ct, k, m, r, g):
    sr = math.ceil(math.log2(k)) + math.ceil(r * (m - math.ceil(math.log2(k))) / g)
    if r == 1:
        sr1 = math.ceil(math.log2(k))
    else:
        sr1 = math.ceil(math.log2(k)) + math.ceil((r - 1) * (m - math.ceil(math.log2(k))) / g)
    s_len = sr - sr1
    # print(k, r, sr, sr1, s_len)
    sl = int(math.pow(2, s_len))
    l = k * sl
    # s = np.zeros(sl)
    # s = [0] * sl
    s = [0 for col in range(sl)]
    for i in range(sl):
        s[i] = i
    # print(sl, s)
    # D = np.zeros(l)
    # D = [0] * l
    D = [0 for col in range(l)]
    # p+1应该比i*sl+j计算的快吧，用p来指示到哪一位了
    p = 0
    for i in range(k):
        for j in range(sl):
            # print(int(Ct[i]), int(s[i]))
            t_num = '{:b}'.format(int(Ct[i])) + '{:b}'.format(int(s[j])).zfill(s_len)
            D[p] = int(t_num, 2)
            # print(t_num, D[p])
            # print(D[p])
            p += 1
    # print(Ct[k - 1], s[sl - 1], sl, D[l - 1])
    # print(D[0], D[l - 1])
    return D


def construct_D_file(Ct, k, m, r, g):
    sr = math.ceil(math.log2(k)) + math.ceil(r * (m - math.ceil(math.log2(k))) / g)
    if r == 1:
        sr1 = math.ceil(math.log2(k))
    else:
        sr1 = math.ceil(math.log2(k)) + math.ceil((r - 1) * (m - math.ceil(math.log2(k))) / g)
    s_len = sr - sr1
    sl = int(math.pow(2, s_len))
    # l = k * sl
    # s = np.zeros(sl)
    # s = [0] * sl
    s = [0 for col in range(sl)]
    for i in range(sl):
        s[i] = i
    # D = np.zeros(l)
    # 将D存入文件中
    filename = '../../temp/olh_sample/D' + str(r) + '.txt'
    f1 = open(filename, 'w')
    # p+1应该比i*sl+j计算的快吧，用p来指示到哪一位了
    # p = 0
    for i in range(k):
        for j in range(sl):
            # print(int(Ct[i]), int(s[i]))
            t_num = '{:b}'.format(int(Ct[i])) + '{:b}'.format(int(s[j])).zfill(s_len)
            f1.write(str(int(t_num, 2)))
            # f1.write(str(float(int(t_num, 2))))
            f1.write('\n')
            # D[p] = int(t_num, 2)
            # print(D[p])
            # p += 1
    # print(Ct[k - 1], s[sl - 1], sl, D[l - 1])
    return 0


def construct_C(C1, k, D):
    # top_k_data = np.zeros(k)
    # top_k_dist = np.zeros(k)
    # top_k_data = [0] * k
    # top_k_dist = [0] * k
    top_k_data = [0 for col in range(k)]
    top_k_dist = [0 for col in range(k)]
    for i in range(k):
        # print(C1[C1.argmax()], C1.argmax(), D[C1.argmax()])
        top_k_dist[i] = max(C1)
        top_k_data[i] = D[C1.index(max(C1))]
        C1[C1.index(max(C1))] = -100
        # print(C1[C1.argmax()])
    return top_k_data, top_k_dist


def construct_C_file(k, r):
    # top_k_data = np.zeros(k)
    # top_k_dist = np.zeros(k)
    # top_k_data = [0] * k
    # top_k_dist = [0] * k
    top_k_data = [0 for col in range(k)]
    top_k_dist = [0 for col in range(k)]
    file_d_name = '../../temp/olh_sample/D' + str(r) + '.txt'
    file_dist_name = '../../temp/olh_sample/dist' + str(r) + '.txt'
    file_d = open(file_d_name, 'r')
    file_dist = open(file_dist_name, 'r')
    str_data = file_d.readline().replace('\n', '')
    str_dist = file_dist.readline().replace('\n', '')
    # str_dist = file_dist.readline()
    # str_data = file_d.readline()
    while str_dist:
        dist = float(str_dist)
        data = int(str_data)
        t_min = min(top_k_dist)
        # t_min = top_k_dist.min()
        if dist > t_min:
            p_min = top_k_dist.index(min(top_k_dist))
            # p_min = top_k_dist.argmin()
            top_k_dist[p_min] = dist
            top_k_data[p_min] = data
        str_dist = file_dist.readline()
        str_data = file_d.readline()
    file_d.close()
    file_dist.close()
    return top_k_data, top_k_dist


def cal_f1(ct, cg):
    c = []
    for i in range(len(ct)):
        if ct[i] in cg:
            c.append(ct[i])
    lc = len(c)
    p = lc / len(ct)
    r = lc / len(cg)
    if p == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    print(c)
    return f1


def cal_ncr(ct, cg, dist_ct, dist_cg, k):
    dist_t1 = [0 for col in range(k)]
    dist_t2 = [0 for col in range(k)]
    for i in range(k):
        dist_t1[i] = dist_ct[i]
        dist_t2[i] = dist_cg[i]
    order_ct = []
    order_cg = []
    ncr = 0
    for i in range(k):
        t_index1 = dist_t1.index(max(dist_t1))
        order_ct.append(ct[t_index1])
        dist_t1[t_index1] = 0
        t_index2 = dist_t2.index(max(dist_t2))
        order_cg.append(cg[t_index2])
        dist_t2[t_index2] = 0
    for i in range(k):
        t = order_ct[i]
        if t in order_cg:
            ncr = ncr + (k - order_cg.index(t))
    ncr = ncr / (k * (k + 1) / 2)
    return ncr


def PEM_OLH(prefix_X, k, m, g, c, epsilon, N):
    S0 = math.ceil(math.log2(k))
    C0 = [x for x in range(0, pow(2, S0))]
    l_s = math.ceil(math.log2(k)) + math.ceil((m - math.ceil(math.log2(k))) / g)
    # print(l_s)
    if l_s < 20:
    # if l_s < 1:
        # 对应1->g轮
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            # prefix_X, Ni, si = data_to_prefix(X, N, i, g, c, m)
            # print(prefix_X)
            if i == 1:
                Ct = C0
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_OlH = OLH(t_X, Ni, c, epsilon, D)
                Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
                # print('-------------------------------------------------------------')
                # print(Ct)
                # print(OLH_dist)
            else:
                # Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_OlH = OLH(t_X, Ni, c, epsilon, D)
                Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
                # print('-------------------------------------------------------------')
                # print(Ct)
                # print(OLH_dist)
    else:
        # 数据量过大，要使用文件存储频率估计的结果
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            # prefix_X, Ni, si = data_to_prefix(X, N, i, g, c, m)
            # print(prefix_X)
            if i == 1:
                Ct = C0
                construct_D_file(Ct, k, m, i, g)
                OLH_file(t_X, Ni, c, epsilon, i)
                Ct, OLH_dist = construct_C_file(k, i)
                # print(Ct)
                # print(OLH_dist)
            else:
                # Ct, OLH_dist = construct_C_file(k, i)
                construct_D_file(Ct, k, m, i, g)
                OLH_file(t_X, Ni, c, epsilon, i)
                Ct, OLH_dist = construct_C_file(k, i)
                # print(Ct)
                # print(OLH_dist)
    # f1_olh = cal_f1(Real_k_data, Ct)
    return Ct, OLH_dist


def PEM_OLH_sample(prefix_X, k, m, g, c, epsilon, N):
    S0 = math.ceil(math.log2(k))
    C0 = [x for x in range(0, pow(2, S0))]
    l_s = math.ceil(math.log2(k)) + math.ceil((m - math.ceil(math.log2(k))) / g)
    # print(l_s)
    if l_s < 20:
    # if l_s < 1:
        # 对应1->g轮
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            # prefix_X, Ni, si = data_to_prefix(X, N, i, g, c, m)
            # print(prefix_X)
            if i == 1:
                Ct = C0
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_OlH_sample = OLH_sample(t_X, Ni, c, epsilon, D)
                Ct, OLH_sample_dist = construct_C(EstimateDist_OlH_sample, k, D)
                # print('-------------------------------------------------------------')
                # print(Ct)
                # print(OLH_dist)
            else:
                # Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_OlH_sample = OLH_sample(t_X, Ni, c, epsilon, D)
                Ct, OLH_sample_dist = construct_C(EstimateDist_OlH_sample, k, D)
                # print('-------------------------------------------------------------')
                # print(Ct)
                # print(OLH_dist)
    else:
        # 数据量过大，要使用文件存储频率估计的结果
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            # prefix_X, Ni, si = data_to_prefix(X, N, i, g, c, m)
            # print(prefix_X)
            if i == 1:
                Ct = C0
                construct_D_file(Ct, k, m, i, g)
                OLH_sample_file(t_X, Ni, c, epsilon, i)
                Ct, OLH_sample_dist = construct_C_file(k, i)
                # print(Ct)
                # print(OLH_dist)
            else:
                # Ct, OLH_dist = construct_C_file(k, i)
                construct_D_file(Ct, k, m, i, g)
                OLH_sample_file(t_X, Ni, c, epsilon, i)
                Ct, OLH_sample_dist = construct_C_file(k, i)
                # print(Ct)
                # print(OLH_dist)
    # f1_olh = cal_f1(Real_k_data, Ct)
    return Ct, OLH_sample_dist


def PEM_wheel(prefix_X, k, m, g, c, epsilon, N):
    S0 = math.ceil(math.log2(k))
    C0 = [x for x in range(0, pow(2, S0))]
    l_s = math.ceil(math.log2(k)) + math.ceil((m - math.ceil(math.log2(k))) / g)
    if l_s < 20:
    # if l_s < 1:
        # 对应1->g轮
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            # prefix_X, Ni, si = data_to_prefix(X, N, i, g, c, m)
            # print(prefix_X)
            if i == 1:
                Ct = C0
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_wheel = set_wheel(t_X, Ni, c, epsilon, D)
                Ct, wheel_dist = construct_C(EstimateDist_wheel, k, D)
                # EstimateDist_OlH = OLH(t_X, Ni, c, epsilon, D)
                # Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
                # print('-------------------------------------------------------------')
                # print(Ct)
                # print(OLH_dist)
            else:
                # Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_wheel = set_wheel(t_X, Ni, c, epsilon, D)
                Ct, wheel_dist = construct_C(EstimateDist_wheel, k, D)
                # EstimateDist_OlH = OLH(t_X, Ni, c, epsilon, D)
                # Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
                # print('-------------------------------------------------------------')
                # print(Ct)
                # print(OLH_dist)
    else:
        for i in range(1, g + 1):
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            # prefix_X, Ni, si = data_to_prefix(X, N, i, g, c, m)
            # print(prefix_X)
            if i == 1:
                Ct = C0
                construct_D_file(Ct, k, m, i, g)
                set_wheel_file(t_X, Ni, c, epsilon, i)
                Ct, wheel_dist = construct_C_file(k, i)
                # print(Ct)
                # print(OLH_dist)
            else:
                # Ct, OLH_dist = construct_C_file(k, i)
                construct_D_file(Ct, k, m, i, g)
                set_wheel_file(t_X, Ni, c, epsilon, i)
                Ct, wheel_dist = construct_C_file(k, i)
    # f1_olh = cal_f1(Real_k_data, Ct)
    return Ct, wheel_dist


def PEM_PCKV_UE(prefix_X, k, m, g, c, epsilon, N, si):
    S0 = math.ceil(math.log2(k))
    C0 = [x for x in range(0, pow(2, S0))]
    l_s = math.ceil(math.log2(k)) + math.ceil((m - math.ceil(math.log2(k))) / g)
    if l_s < 20:
    # if l_s < 1:
        # 对应1->g轮
        for i in range(1, g + 1):
            # 目前能取到的最大值加1，即PCKV-UE虚假值的起点
            s = pow(2, si[i])
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            # prefix_X, Ni, si = data_to_prefix(X, N, i, g, c, m)
            # print(prefix_X)
            if i == 1:
                Ct = C0
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_PCKV_UE = PCKV_UE(t_X, Ni, c, epsilon, D, s)
                Ct, PCKV_UE_dist = construct_C(EstimateDist_PCKV_UE, k, D)
                # EstimateDist_OlH = OLH(t_X, Ni, c, epsilon, D)
                # Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
                # print('-------------------------------------------------------------')
                # print(Ct)
                # print(OLH_dist)
            else:
                # Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
                D = construct_D(Ct, k, m, i, g)
                EstimateDist_PCKV_UE = PCKV_UE(t_X, Ni, c, epsilon, D, s)
                Ct, PCKV_UE_dist = construct_C(EstimateDist_PCKV_UE, k, D)
                # EstimateDist_OlH = OLH(t_X, Ni, c, epsilon, D)
                # Ct, OLH_dist = construct_C(EstimateDist_OlH, k, D)
                # print('-------------------------------------------------------------')
                # print(Ct)
                # print(OLH_dist)
    else:
        for i in range(1, g + 1):
            s = pow(2, si[i]) - 1
            t_X, Ni = split_prefix_X(prefix_X, i, g, N)
            # prefix_X, Ni, si = data_to_prefix(X, N, i, g, c, m)
            # print(prefix_X)
            if i == 1:
                Ct = C0
                construct_D_file(Ct, k, m, i, g)
                PCKV_UE_file(t_X, Ni, c, epsilon, i, s)
                Ct, PCKV_UE_dist = construct_C_file(k, i)
                # print(Ct)
                # print(OLH_dist)
            else:
                # Ct, OLH_dist = construct_C_file(k, i)
                construct_D_file(Ct, k, m, i, g)
                PCKV_UE_file(t_X, Ni, c, epsilon, i, s)
                Ct, PCKV_UE_dist = construct_C_file(k, i)
    # f1_olh = cal_f1(Real_k_data, Ct)
    return Ct, PCKV_UE_dist


# N是用户数量，d是数据域的大小，m是数据域大小二进制表示的长度，
# c是用户手里数据的条数，epsilon是隐私预算，g是分的组数，k是找top-k
if __name__ == '__main__':
    # N = 100000
    m = 160
    # c = 10
    # k = 16
    g = 16
    epsilon = 2
    # a = np.zeros(d)
    # X, Real_k_dist, Real_k_data = generate_data(N, c, k)
    file_name = '../../data/url_data/160_10.txt'
    f1_name = '../../result/real_url/k/f1_tolh_sample_g16_k.txt'
    ncr_name = '../../result/real_url/k/ncr_tolh_sample_g16_k.txt'
    file_f1 = open(f1_name, 'w')
    file_ncr = open(ncr_name, 'w')
    ct_name = '../../result/real_url/k/ct_tolh_sample_g16_k.txt'
    dist_name = '../../result/real_url/k/dist_tolh_sample_g16_k.txt'
    file_ct = open(ct_name, 'w')
    file_dist = open(dist_name, 'w')
    t_num = 4
    for t_num_i in range(t_num):
        num = 16
        olh_sample_f1 = [0 for col in range(num)]
        olh_sample_ncr = [0 for col in range(num)]
        for i in range(num):
            k = i * 2 + 1
            X, Real_k_dist, Real_k_data, N, c = read_data(file_name, k)
            # print(k)
            # print(Real_k_data)
            # print(Real_k_dist)
            prefix_X, si = data_to_prefix(X, N, g, c, m, k)
            OLH_sample_Ct, OLH_sample_dist = PEM_OLH_sample(prefix_X, k, m, g, c, epsilon, N)
            olh_sample_f1[i] = cal_f1(Real_k_data, OLH_sample_Ct)
            olh_sample_ncr[i] = cal_ncr(Real_k_data, OLH_sample_Ct, Real_k_dist, OLH_sample_dist, k)
            # print(i, 'f1', olh_sample_f1[i])
            # print(i, 'ncr', olh_sample_ncr[i])
            print(k, 'f1', olh_sample_f1[i], 'ncr', olh_sample_ncr[i])
            print(OLH_sample_Ct)
            print(OLH_sample_dist)
            file_f1.write(str(olh_sample_f1[i]) + ' ')
            file_ncr.write(str(olh_sample_ncr[i]) + ' ')
            for t_file in range(k):
                file_ct.write(str(OLH_sample_Ct[t_file]) + ' ')
                file_dist.write(str(OLH_sample_dist[t_file]) + ' ')
            file_ct.write('\n')
            file_dist.write('\n')
        file_ncr.write('\n')
        file_f1.write('\n')
        print('OLH_sample', t_num_i)
        print(olh_sample_f1)
        print(olh_sample_ncr)
    file_f1.close()
    file_ncr.close()
    file_ct.close()
    file_dist.close()

