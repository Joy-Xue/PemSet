import math
from scipy.special import comb
from _functools import reduce
import matplotlib.pyplot as plt
import numpy as np


# def cal_below_t(n, m, t):
#     sum = 0
#     for i in range(0, math.floor(m / (t + 1)) + 1):
#         sum += pow(-1, i) * comb(n, i) * comb(n + m - 1 - i * (t + 1), m - i * (t + 1))
#         # print(i, sum)
#     # print(sum)
#     return sum
#
#
# def cal_t(n, m, t):
#     if t == 1:
#         return cal_below_t(n, m, t)
#     else:
#         # print(cal_below_t(n, m, t), cal_below_t(n, m, t - 1))
#         return cal_below_t(n, m, t) - cal_below_t(n, m, t - 1)
#
#
# def Hnm(n, m):
#     return comb(n + m - 1, m)
#
#
# OverflowError: int too large to convert to float
# c太大时候也不行
def cal_factorial(a, b):
    if b >= a:
        return reduce(lambda x, y: x*y, range(a, b + 1))
    else:
        return 1


def cal_below_t2(n, m, t):
    sum = 0
    for i in range(0, math.floor(m / (t + 1)) + 1):
        # sum += pow(-1, i) * cal_factorial(m - i * (t + 1) + 1, m) * cal_factorial(n - i + 1, n) / cal_factorial(1, i) / cal_factorial(n + m - i * (t + 1), n + m - 1)
        # print(i, sum)
        t_num = pow(-1, i) * cal_factorial(m - i * (t + 1) + 1, m) / cal_factorial(1, i)
        for j in range(0, i):
            t_num = t_num * (n - j) / (n + m - 1 - j)
        # t_num = t_num / (n - i)
        for j in range(1, i * t + 1):
            t_num = t_num / (n + m - i - j)
        sum += t_num
    # print(sum)
    return sum


# 第r轮应该flatten的次数
def cal_r_flatten_num(m, k, c, g, r):
    si = pow(2, math.ceil(math.log2(k)) + math.ceil(r * (m - math.ceil(math.log2(k))) / g))
    # print(math.ceil(math.log2(k)) + math.ceil(i * (m - math.ceil(math.log2(k))) / g))
    # sum = Hnm(si, c)
    portion = [0 for col in range(c)]
    t_portion = [0 for col in range(c)]
    for i in range(c):
        t = 1 + i
        portion[i] = cal_below_t2(si, c, t)
        # t_portion[i] = cal_below_t(si, c, t) / sum
    # print(portion)
    t_portion = 0
    i = 0
    while t_portion < 0.995:
        t_portion = portion[i]
        i += 1
    # print(i, portion)
    return i
# m = 160
# g = 16
# k = 16
# c = 100
# # print(comb(pow(2, 64) + m - 1, m))
# for i in range(1, g):
#     si = pow(2, math.ceil(math.log2(k)) + math.ceil(i * (m - math.ceil(math.log2(k))) / g))
#     # print(math.ceil(math.log2(k)) + math.ceil(i * (m - math.ceil(math.log2(k))) / g))
#     # sum = Hnm(si, c)
#     portion = [0 for col in range(c)]
#     t_portion = [0 for col in range(c)]
#     for i in range(c):
#         t = 1 + i
#         portion[i] = cal_below_t2(si, c, t)
#         # t_portion[i] = cal_below_t(si, c, t) / sum
#     print(portion)
#     # print(t_portion)
