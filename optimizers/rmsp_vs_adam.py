# -*- coding: utf-8 -*-
# @File    : rmsp_vs_adam.py
# @Author  : Gaoli Chen
# @Time    : 2021/08/12
# @Desc    :

import numpy as np
import math

dim = 2
N = 10000
noise_size = 0.1
center = 1 + np.random.random((dim,))
noise = noise_size * (np.random.random((N, dim)) - 0.5)

def f(x, i):
    return np.dot(x - center - noise[i], x - center - noise[i])


def f_derivative(x, i):
    return 2 * (x - center - noise[i])

class Adam(object):
    def __init__(self, beta1, beta2, alpha):
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.accuracy = 1e-8


    def __call__(self, fun, fun_der):
        m, v = 0.0, 0.0
        x = [.0] * dim
        for i in range(N):
            g = fun_der(x, i)
            g2 = np.multiply(g, g)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g2
            m_hat = m / (1 - self.beta1)
            v_hat = np.sqrt(v / (1 - self.beta2))
            delta = self.alpha * m_hat / (v_hat + 1e-8)
            x -= delta
            if i % 1000 == 0:
                print('i=', i, 'delta=', math.sqrt(np.dot(delta, delta)), 'f(x)=', fun(x, i))
            if math.sqrt(np.dot(delta, delta)) < self.accuracy:
                break

        return x, i

class RMSProp(object):
    def __init__(self, rho, step_size):
        self.rho = rho
        self.accuracy = 1e-8
        self.step_size = step_size

    def __call__(self, fun, fun_der):
        s = .0
        x = [.0] * dim
        for i in range(N):
            g = fun_der(x, i)
            s = s * self.rho + (1 - self.rho) * np.multiply(g, g)
            delta = self.step_size * g / (1e-8 + np.sqrt(s))
            x -= delta
            if i % 1000 == 0:
                print('i=', i, 'delta=', math.sqrt(np.dot(delta, delta)), 'f(x)=', fun(x, i))
            if math.sqrt(np.dot(delta, delta)) < self.accuracy:
                break
            
        return x, i

def compare_rmsp_adam():
    adam = Adam(beta1 = 0.9, beta2 = 0.999, alpha = 1e-3)
    res1, step1 = adam(f, f_derivative)
    
    rmsp = RMSProp(rho = 0.9, step_size = 1e-3)
    res2, step2 = rmsp(f, f_derivative)

    error1 = res1 - center
    error2 = res2 - center

    print('noise mean=', np.mean(noise), 'sigma=', np.std(noise))
    print('adam: target=', center, 'res=', res1, 'error=', error1, '|error1|=', math.sqrt(np.dot(error1, error1)), 'step1 =', step1)
    print('rmsp: target=', center, 'res=', res2, 'error=', error2, '|error2|=', math.sqrt(np.dot(error2, error2)), 'step2 =', step2)


if __name__ == "__main__":
    compare_rmsp_adam()
