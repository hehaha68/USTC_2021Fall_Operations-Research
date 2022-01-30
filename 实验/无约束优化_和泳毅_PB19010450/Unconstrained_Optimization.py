# -*- coding: utf-8 -*-
""" 
@Time    : 2021/12/28 17:20
@Author  : 和泳毅 PB19010450
@FileName: Unconstrained_Optimization.py
@SoftWare: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
import math


def Object1(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2


def Grad1(x):
    g = np.zeros((2, 1))
    g[0, 0] = 400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1)
    g[1, 0] = -200 * (x[0] ** 2 - x[1])
    return g


def H1(x):
    return np.matrix([[1200 * x[0] * x[0] - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])


def Object2(x):
    return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
            2.625 - x[0] + x[0] * x[1] ** 3) ** 2


def Grad2(x):
    g = np.zeros((2, 1))
    g[0, 0] = 2 * (x[1] - 1) * (1.5 - x[0] + x[0] * x[1]) + 2 * (x[1] - 1) * (2.25 - x[0] + x[0] * x[1] ** 2) + \
              2 * (x[1] ** 3 - 1) * (2.625 - x[0] + x[0] * x[1] ** 3)
    g[1, 0] = 2 * (1.5 - x[0] + x[0] * x[1]) + 4 * x[0] * x[1] * (2.25 - x[0] + x[0] * x[1] ** 2) + \
              (6 * x[0] * x[1] ** 2) * (2.625 - x[0] + x[0] * x[1] ** 3)
    return g


def H2(x):
    return np.matrix([[2 * (x[1] - 1) ** 2 + 2 * (x[1] ** 2 - 1) ** 2 + 2 * (x[1] ** 3 - 1) ** 2,
                       3 + 9 * x[1] + 15.75 * x[1] ** 2 + x[0] * (
                               -4 - 4 * x[1] - 12 * x[1] ** 2 + 8 * x[1] ** 3 + 12 * x[1] ** 5)],
                      [3 + 9 * x[1] + 15.75 * x[1] ** 2 + x[0] * (
                              -4 - 4 * x[1] - 12 * x[1] ** 2 + 8 * x[1] ** 3 + 12 * x[1] ** 5),
                       x[0] * (9 + 31.5 * x[1] + x[0] * (-2 - 12 * x[1] + 12 * x[1] ** 2 + 30 ** 4))]])


def Object3(x):
    return np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1


def Grad3(x):
    g = np.zeros((2, 1))
    g[0, 0] = -1.5 + 2 * (x[0] - x[1]) + np.cos(x[0] + x[1])
    g[1, 0] = 2.5 - 2 * (x[0] - x[1]) + np.cos(x[0] + x[1])
    return g


def H3(x):
    return np.matrix(
        [[2 - np.sin(x[0] + x[1]), -2 - np.sin(x[0] + x[1])], [2 - np.sin(x[0] + x[1]), -2 - np.sin(x[0] + x[1])]])


class UOM():
    def __init__(self, Object, Grad, H, epochs, Lambda):
        self.epochs = epochs
        self.Lambda = Lambda
        self.object = Object
        self.grad = Grad
        self.H = H

    def Wolfe_Powell(self, x, alpha_, f, g, d, rho, sigma):
        epochs = 1000
        a1, a2 = 0, alpha_
        alpha = (a1 + a2) / 2
        phi0, phi0_ = f, np.dot(g.T[0], d)[0]
        phi1, phi1_ = phi0, phi0_
        k = 0
        for i in range(epochs):
            k += 1
            phi = self.object(x + alpha * d.T[0])
            if phi <= phi0 + rho * alpha * phi0_:
                g_new = self.grad(x + alpha * d.T[0])
                phi_ = np.dot(g_new.T[0], d)[0]
                if phi_ >= sigma * phi0_:
                    break
                else:
                    if abs(phi1_ - phi_) < 1e-32:
                        break
                    alpha_new = alpha + (alpha - a1) * phi_ / (phi1_ - phi_)
                    a1 = alpha
                    alpha = alpha_new
                    phi1 = phi
                    phi1_ = phi_
            else:
                alpha_new = a1 + 0.5 * (a1 - alpha) ** 2 * phi1_ / ((phi1 - phi) - (a1 - alpha) * phi1_)
                a2 = alpha
                alpha = alpha_new
        return alpha

    def Gold_div_search(self, x, a, b, esp, d):
        # 黄金分割法一维搜索，返回步长
        rou = 1 - (np.sqrt(5) - 1) / 2  # 1-rou为黄金分割比
        alpha1 = a + rou * (b - a)
        alpha2 = b - rou * (b - a)
        while b - a > esp:
            x1 = x + alpha1 * d.T[0]
            x2 = x + alpha2 * d.T[0]
            f1 = self.object(x1)
            f2 = self.object(x2)
            if f1 > f2:  # 如果f1>f2，则在区间(alpha1,b)内搜索
                a = alpha1
                alpha1 = alpha2
                alpha2 = b - rou * (b - a)
            elif f1 < f2:  # 如果f1<f2,则在区间(a,alpha2)内搜索
                b = alpha2
                alpha2 = alpha1
                alpha1 = a + rou * (b - a)
            else:  # 如果f1=f2，则在区间(alpha1,alpha2)内搜索
                a = alpha1
                b = alpha2
                alpha1 = a + rou * (b - a)
                alpha2 = b - rou * (b - a)
        return a

    def Quasi_Newton(self, x0, method):
        L = np.zeros(self.epochs + 1)
        x = np.zeros((self.epochs + 1, len(x0)))
        Hk = np.eye(len(x0))
        x[0] = x0
        L[0] = self.object(x[0])
        epochs = 0
        for i in range(self.epochs):
            gk = self.grad(x[i])
            dk = -1.0 * np.dot(Hk, gk)
            # alpha = self.Gold_div_search(x=x[i],a=0,b=0.5,esp=10e-16,d=dk)
            alpha = self.Wolfe_Powell(x=x[i], alpha_=1, f=L[i], g=gk, d=dk, rho=0.25, sigma=0.5)
            x[i + 1] = x[i] + alpha * dk.T[0]

            L[i + 1] = self.object(x[i + 1])
            if np.abs(L[i + 1] - L[i]) <= self.Lambda:
                break

            sk = x[i + 1] - x[i]
            sk = sk.reshape((2, 1))
            yk = self.grad(x[i + 1]) - gk
            if method == 'DFP':
                if np.dot(sk.T, yk) > 0:
                    Hyy = np.dot(np.dot(Hk, yk), yk.T)
                    sy = np.dot(sk.T, yk)
                    yHy = np.dot(np.dot(yk.T, Hk), yk)
                    Hk = Hk - np.dot(Hyy, Hk) / yHy + np.dot(sk, sk.T) / sy
            elif method == 'BFGS':
                Hy = np.dot(Hk, yk)
                sy = np.dot(sk.T, yk)
                syt = np.dot(sk, yk.T)
                yHy = np.dot(np.dot(yk.T, Hk), yk)
                Hk = Hk + (1 + yHy / sy) * np.dot(sk, sk.T) / sy - (np.dot(Hy, sk.T) + np.dot(syt, Hk)) / sy
            epochs = i + 1
        return epochs, x, L

    def Newton(self, x0):
        L = np.zeros(self.epochs + 1)
        x = np.zeros((self.epochs + 1, len(x0)))
        x[0] = x0
        L[0] = self.object(x[0])
        epochs = 0
        for i in range(self.epochs):
            H = self.H(x[i])
            if np.linalg.matrix_rank(H) < 2:
                print("Hesse矩阵奇异！")
                return epochs, x, L
            Hk = H.I
            gk = self.grad(x[i])
            dk = -1.0 * np.dot(Hk, gk)
            x[i + 1] = x[i] + dk.T[0]

            L[i + 1] = self.object(x[i + 1])
            if np.abs(L[i + 1] - L[i]) <= self.Lambda:
                break

            epochs = i + 1
        return epochs, x, L

    def Gradient(self, x0):
        epochs = 0
        L = np.zeros(self.epochs + 1)
        x = np.zeros((self.epochs + 1, len(x0)))
        x[0] = x0
        L[0] = self.object(x[0])
        for i in range(self.epochs):
            gk = self.grad(x[i])
            dk = -1.0 * gk
            alpha = self.Gold_div_search(x=x[i], a=0, b=0.5, esp=10e-16, d=dk)
            x[i + 1] = x[i] + alpha * dk.T[0]

            L[i + 1] = self.object(x[i + 1])
            if np.abs(L[i + 1] - L[i]) <= self.Lambda:
                break

            epochs = i + 1
        return epochs, x, L


if __name__ == "__main__":
    f = input("请输入需要测试的函数：\n【1】Rosenbrock\n【2】Beale\n【3】McCormick\n")
    method = input("请输入需要使用的方法:\n"
                   "【1】采用精确一维搜索的最速下降法\n"
                   "【2】经典牛顿法\n"
                   "【3】采用基于Wolfe-Powell非精确一维搜索的DFP法\n"
                   "【4】采用基于Wolfe-Powell非精确一维搜索的BFGS法\n")
    X0 = input("请输入初始点：").split()
    flag = True
    if f == '1':
        Object = Object1
        Grad = Grad1
        H = H1
    elif f == '2':
        Object = Object2
        Grad = Grad2
        H = H2
    elif f == '3':
        Object = Object3
        Grad = Grad3
        H = H3
    else:
        flag = False
        print("函数输入有误！\n")
    if method not in ['1', '2', '3', '4']:
        flag = False
        print("方法输入有误！\n")
    if flag:
        x0 = [0, 0]
        x0[0], x0[1] = float(X0[0]), float(X0[1])
        uom = UOM(Object, Grad, H, epochs=500, Lambda=10e-16)
        if method == '1':
            epochs, x, L = uom.Gradient(x0)
        elif method == '2':
            epochs, x, L = uom.Newton(x0)
        elif method == '3':
            epochs, x, L = uom.Quasi_Newton(x0, method='DFP')
        else:
            epochs, x, L = uom.Quasi_Newton(x0, method='BFGS')

        print("迭代{}次".format(epochs))
        print("最优解：x1={},x2={}".format(x[epochs][0], x[epochs][1]))
        print("最优值：f={}".format(L[epochs]))