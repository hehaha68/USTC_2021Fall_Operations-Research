import numpy as np
from matplotlib import pyplot as plt


def gradient(x):
    n = x.shape[0]
    result = np.zeros((n, 1))
    result[0] = -400 * x[0] * (x[1] - x[0] ** 2) + 2 * (x[0] - 1)
    if n == 2:
        result[1] = 200 * (x[1] - x[0] ** 2)
    elif n == 3:
        result[1] = 200 * (x[1] - x[0] ** 2) - 400 * x[1] * (x[2] - x[1] ** 2) + 2 * (x[1] - 1)
        result[2] = 200 * (x[2] - x[1] ** 2)
    return result


def f(x):
    if x.shape[0] == 2:
        return 100 * ((x[1] - x[0] ** 2) ** 2) + (x[0] - 1) ** 2
    elif x.shape[0] == 3:
        return 100 * ((x[1] - x[0] ** 2) ** 2) + (x[0] - 1) ** 2 + (x[1] - 1) ** 2 + 100 * (x[2] - x[1] ** 2) ** 2


def Wolfe_Powell(x, d):
    # 初始化
    step, limit = 0, 1000
    rho, sigma = 0.15, 0.5
    Alpha = 2
    a1, a2 = 0, Alpha
    alpha = (a1 + a2) / 2
    phi1 = f(x)
    phi1_ = gradient(x).T @ d
    while step < limit:
        step += 1
        phi = f(x + alpha * d)
        if phi <= phi1 + rho * alpha * phi1_:  # condition1
            phi_ = gradient(x + alpha * d).T @ d
            if phi_ >= sigma * phi1_:  # condition2
                return alpha

            else:  # 外插法
                new = alpha + phi_ * (alpha - a1) / (phi1_ - phi_)
                a1 = alpha
                alpha = new
                phi1 = phi
                phi1_ = phi_

        else:  # 内插法
            new = a1 - 0.5 * phi1_ * (alpha - a1) ** 2 / (phi - phi1 - (alpha - a1) * phi1_)
            a2 = alpha
            alpha = new

    return alpha


def Quasi_Newton(x, method):
    # 初始化
    epsilon, limit = 1e-5, 2000
    n = x.shape[0]
    H = np.eye(n)
    step = 1
    g0 = gradient(x)
    d = -H @ g0
    Y = [f(x)]
    I = np.eye(n)
    while step < limit and np.sqrt(np.sum(g0 ** 2)) > epsilon:
        alpha = Wolfe_Powell(x, d)
        delta = alpha * d
        x += delta
        g1 = gradient(x)
        y = g1 - g0
        if method == '1':
            H += (delta @ delta.T) / (delta.T @ y) - (H @ y @ y.T @ H) / (y.T @ H @ y)
        else:
            H = (I - delta @ y.T / (delta.T @ y)) @ H @ (I - delta @ y.T / (delta.T @ y)).T + delta @ delta.T / (
                    delta.T @ y)
        d = -H @ gradient(x)
        g0 = g1
        Y.append(f(x))
        step += 1
    print('x = ', x.tolist())
    print('▽f(x) = ', gradient(x).tolist())
    print('f(x) = ', f(x))
    print('迭代次数为', step)
    X = range(step)
    plt.xlabel('Iteration steps')
    plt.ylabel('f(x)')
    plt.plot(X, Y)
    plt.show()


def main():
    print('-' * 3, '程序名称：非精确一维搜索求解无约束最优化', '-' * 3)
    print('-' * 6, '王湘峰PB19030861', '-' * 6)
    v = input('请输入初始点坐标（以空格作为分割）\n').split()
    x = np.zeros((len(v), 1))
    for i in range(len(v)):
        x[i] = v[i]
    method = input('请选择下降方法：1 → DFP 2 → BFGS\n')
    while True:
        if method == '1' or method == '2':
            break
        else:
            print('非法输入')
            method = input('请选择下降方法：1 → DFP 2 → BFGS\n')
    Quasi_Newton(x, method)


if __name__ == '__main__':
    main()
