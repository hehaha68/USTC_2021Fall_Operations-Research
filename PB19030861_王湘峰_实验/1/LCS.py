from time import time


def LCS(s1, s2):
    start = time()
    n1 = len(s1)
    n2 = len(s2)
    table = [[0] * (n2 + 1) for i in range(n1 + 1)]
    record = [[0] * n2 for i in range(n1)]
    for i in range(n1):
        for j in range(n2):
            if s1[i] == s2[j]:
                table[i + 1][j + 1] = table[i][j] + 1
                record[i][j] = 1
            elif table[i][j + 1] > table[i + 1][j]:
                table[i + 1][j + 1] = table[i][j + 1]
                record[i][j] = 2
            else:
                table[i + 1][j + 1] = table[i + 1][j]
                record[i][j] = 3
    # print longest common string
    lcs = []
    i, j = n1 - 1, n2 - 1
    while True:
        if i < 0 or j < 0:
            break
        if record[i][j] == 1:
            lcs.insert(0, s1[i])
            i -= 1
            j -= 1
        elif record[i][j] == 2:
            i -= 1
        else:
            j -= 1
    print('\n最长公共子序列为：', end='')
    for s in lcs:
        print(s, end='')
    stop = time()
    print('\n长度为', table[n1][n2], ' 用时：%.16f ms\n' % ((stop - start) * 1000))


if __name__ == '__main__':
    print('-' * 3 + '程序名称：最长公共子序列问题' + '-' * 3)
    print('-' * 5, '王湘峰PB19030861', '-' * 5)
    print('请输入两个字符串，以回车作为分割，输入exit以结束')
    while True:
        s1 = input()
        if s1 == 'exit':
            exit()
        s2 = input()
        LCS(s1, s2)
