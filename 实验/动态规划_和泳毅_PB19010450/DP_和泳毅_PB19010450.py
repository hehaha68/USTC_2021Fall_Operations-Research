# -*- coding: utf-8 -*-
""" 
@Time    : 2022/12/30 21:09
@Author  : 和泳毅 PB19010450
@FileName: DP.py
@SoftWare: PyCharm
"""
# 同顺序M×N排序问题的动态规划方法
import math
import numpy as np
from itertools import combinations

np.set_printoptions(threshold=np.inf)


class T_Set:
    def __init__(self, S, minTime):
        """
        m=2的动态规划方法所用集合T
        :param S: 待排序列
        :param minTime: T(S,t)
        """
        self.S = S
        self.minTime = minTime


def findMinTime(A, Set, t):
    """
    计算T(S,t)
    :param A: 工时矩阵
    :param Set: 零件集合
    :param t: 等待时间
    :return: T(S,t)
    """
    if len(Set.S) == 0:
        return t
    min_time = float("inf")
    for i in range(A.shape[1]):
        Set_ = T_Set(Set.S, Set.minTime)
        if i in Set.S:
            Set_.S.remove(i)
            cur_min_time = A[0][i] + findMinTime(A, T_Set(Set_.S, Set_.minTime), A[1][i] + max(t - A[0][i], 0))
            Set_.S.append(i)
            min_time = min(min_time, cur_min_time)
    return min_time


def DP(A):
    """
    m=2的动态规划方法
    :param A: 工时矩阵
    :return: 最小总工时T(N,0)
    """
    m, n = A.shape
    MAXNUM = 2 ** n
    MAXTIME = 100
    Set = []
    for i in range(MAXNUM):
        Set.append(T_Set([], [0] * (MAXTIME + 2)))
    # 构造全组合
    x = list(range(n))
    k = 0
    for i in range(len(x) + 1):
        for j in combinations(x, i):
            Set[k].S = list(j)
            k += 1
    # 递推
    for i in range(MAXNUM):
        for t in range(MAXTIME):
            Set[i].minTime[t] = findMinTime(A, T_Set(Set[i].S, Set[i].minTime), t)
    return Set[MAXNUM - 1].minTime[0]


def Sum_Time(A, X):
    """
    求解一个可行序列的总工时T(X),即矩阵t_A(X)的最大可行和
    :param A: 工时矩阵
    :param X: 可行序列
    :return: T(X)/最大可行和
    """
    # 求解最优总工时
    if X.shape[0] == 0:
        return 0
    m, n = A[:, X].shape
    A_new = A[:, X]
    f = np.zeros((m + 1, n + 1))
    # 自底向上动态规划求解总工时（工时矩阵最大可行和）
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            f[i][j] = max(f[i][j - 1], f[i - 1][j]) + A_new[i - 1][j - 1]
    return f[m][n]


def Johnson(A):
    """
    m=2的Johnson算法
    :param A: 工时矩阵
    :return: 最优加工顺序
    """
    A = A.astype(np.float16)
    m, n = A.shape
    X = np.zeros(n)
    front, back = 0, n - 1
    # 求解最优排序
    for i in range(n):
        index = np.unravel_index(A.argmin(), A.shape)
        # 如果位于机器1流水线，置于最前
        if index[0] == 0:
            X[front] = index[1]
            front += 1
            A[:, index[1]] = float("inf")
        # 如果位于机器2流水线，置于最后
        else:
            X[back] = index[1]
            back -= 1
            A[:, index[1]] = float("inf")
    X = X.astype(np.int)
    return X


def Bound(A, X, S):
    """
    估计下界B(S...)
    :param A: 工时矩阵
    :param X: 剩余序列
    :param S: 已排的部分序列
    :return: B(S...)
    """
    m, _ = A.shape
    n = X.shape[0]
    R = np.zeros((n, n))
    # 初始化R序列
    for i in range(n):
        R[i] = X
        temp = R[i][i]
        R[i] = np.insert(np.delete(R[i], i), 0, temp)
        R = R.astype(np.int)
    # 估计下界
    b = np.zeros(m)
    for p in range(m - 2):
        min_t = float("inf")
        for i in range(n):
            min_t = min(min_t, Sum_Time(A[p:m - 2], np.array([X[i]])) + Sum_Time(A[m - 2:m], R[i]))
        max_t = max(min_t, A[p, X].sum())
        b[p] = max_t + Sum_Time(A[0:p + 1], S)
    b[m - 2] = Sum_Time(A[m - 2:m], R[0]) + Sum_Time(A[0:m - 1], S)
    b[m - 1] = A[m - 1, R[0]].sum() + Sum_Time(A[0:m], S)
    B = b.max()
    return B


def WSH(A):
    """
    启发式近似解序列
    :param A: 工时矩阵
    :return: 近似解序列
    """
    m, n = A.shape
    L = np.zeros(n)
    for j in range(n):
        a = 0
        for i in range(m):
            a += (i + 1) * A[i][j]
        L[j] = a / A[:,j].sum()
    # 按指标非增的顺序排列所有零件
    X = np.argsort(-L)
    return X


def Delete(X, i):
    X_ = []
    for j in X:
        if j != i:
            X_.append(j)
    return np.array(X_)


def Branch_bound(A, S, upper, leaf):
    """
    分支定界算法
    :param A: 工时矩阵
    :param S: 部分序列
    :param upper: 最优上界
    :param leaf: 叶节点序列及下界值
    :return:
    """
    m, n = A.shape
    X = np.arange(0, n, 1)
    X = np.delete(X, S)
    for i in X:
        X_ = Delete(X, i)
        S_ = S.copy()
        S_.append(i)
        index = Johnson(A[m - 2:m, X_])
        X__ = X_[index]
        lower = Bound(A, X__, np.array(S_))
        # 估计下界大于最优上界则剪枝
        if lower <= upper:
            if X_.shape[0] == 1:
                S_.append(X__[0])
                leaf.append(S_)
            else:
                Branch_bound(A, S_, upper, leaf)
    return np.array(leaf)


def Solution(A, X):
    """
    在叶节点中选取最优解
    :param A: 工时矩阵
    :param X: 分支定界法所得叶节点
    :return: 最优加工顺序,最小总工时
    """
    m, n = X.shape
    Time = np.zeros(m)
    index = []
    for i in range(m):
        Time[i] = Sum_Time(A, X[i])
    min_time = Time.min()
    for i in range(m):
        if Time[i] == min_time:
            index.append(i)
    return X[index, :], min_time


def PrintX(X):
    """
    格式化打印
    :param X: 最优加工顺序
    """
    print("最优加工顺序:")
    for i in range(len(X)):
        print("(" + str(i+1) + ") ", end="")
        for j in range(len(X[i])):
            print("J" + str(X[i][j] + 1), "", end="")
        print("")


if __name__ == "__main__":
    n, m = input("请输入零件个数和机器个数:").split()
    n, m = int(n), int(m)
    A = np.zeros((m, n))
    print("请输入工时矩阵:")
    for i in range(m):
        a = input().split()
        for j in range(n):
            A[i][j] = float(a[j])
    if m == 2:
        print("m=2,可采用动态规划方法和Johnson算法求解：")
        T1 = DP(A)
        print("【1】动态规划方法", "最优总工时：", T1)
        X = Johnson(A)
        T2 = Sum_Time(A, Johnson(A))
        print("【2】Johnson算法", "最优总工时：", T2)
        PrintX([list(X)])
    elif m >= 3:
        print("m>=3,采用结合动态规划的分支定界法求解：")
        X = WSH(A)
        upper = Sum_Time(A, X)
        leaf = Branch_bound(A=A, S=[], upper=upper, leaf=[])
        X, T3 = Solution(A, leaf)
        print("最优总工时：", T3)
        PrintX(X)
    else:
        print("m输入有误!")