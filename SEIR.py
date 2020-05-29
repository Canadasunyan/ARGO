import numpy as np
import matplotlib.pyplot as plt
def SEIR(beta=1, gamma=0.2, sigma=0.5, n=10000, initial=10, days=60):
    # 最初均为易感人群
    array_s = np.zeros(days+1, dtype=np.int)
    array_e = np.zeros(days+1, dtype=np.int)
    array_i = np.zeros(days+1, dtype=np.int)
    array_r = np.zeros(days+1, dtype=np.int)
    array_s[0] = n - initial
    print(array_s)
    array_e[0] = initial
    array_i[0] = array_r[0] = 0
    for index in range(days):
        s = array_s[index]
        print('s= ', s)
        e = array_e[index]
        i = array_i[index]
        r = array_r[index]
        # 计算从i到r的数量
        i2r = min(i, np.random.poisson(int(array_i[index]*gamma), 1)[0])
        r = r + i2r
        i = i - i2r
        # 计算从e到i的数量
        e2i = min(e, np.random.poisson(int(array_e[index]*sigma), 1)[0])
        i = i + e2i
        e = e - e2i
        # 计算从s到e的数量
        s2e = min(s, np.random.poisson(int(array_s[index]*beta*i/n), 1)[0])
        print('s2e: ', s2e)
        e = e + s2e
        s = s - s2e
        # if s < 0:
        #     break
        # 记录更新后的值
        array_s[index+1] = s
        array_e[index+1] = e
        array_i[index+1] = i
        array_r[index+1] = r
    return array_s, array_e, array_i, array_r

if __name__ == "__main__":
    s, e, i, r = SEIR()
    plt.plot(s, label='Susceptible')
    plt.plot(e, label='Exposed')
    plt.plot(i, label='Infectious')
    plt.plot(r, label='Recovered')
    plt.legend(loc='best')
    plt.show()


