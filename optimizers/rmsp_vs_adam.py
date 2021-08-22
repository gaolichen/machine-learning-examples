# -*- coding: utf-8 -*-
# @File    : rmsp_vs_adam.py
# @Author  : Gaoli Chen
# @Time    : 2021/08/12
# @Desc    :

import numpy as np
import math

dim = 2
N = 10000

class Function(object):
    def __init__(self, noise_size = .0, circular = True):
        self.noise_size = noise_size
        self.circular = circular
        self._gen_data()

    def _gen_data(self):
        self.center = 1 + np.random.random((dim,))
        if not self.circular:
            self.axis = np.random.random((dim, )) + 1e-8
        else:
            self.axis = np.ones((dim, ))

        self.noise = (2 * self.noise_size) * (np.random.random((N, dim)) - 0.5)
        print(f'noise mean={np.mean(self.noise)}, noise std={np.std(self.noise)}, {self.axis[0]/self.axis[1]}')

    def __call__(self, x, step):
        v = (x - self.center - self.noise[step]) / self.axis
        return np.dot(v, v)

    def derivative(self, x, step):
        return 2 * (x - self.center - self.noise[step]) / self.axis

class Solver(object):
    def __init__(self, learning_rate, accuracy):
        self.learing_rate = learning_rate
        self.accuracy = accuracy
        
    def __call__(self, function, optimizer):
        x = [.0] * dim
        for i in range(N):
            dx = self.learing_rate * optimizer(function, x, i)
            x -= dx
            
            if math.sqrt(np.dot(dx, dx)) < self.accuracy:
                break

        return x, i + 1

class SGD(object):
    def __call__(self, fun, x, step):
        return fun.derivative(x, step)


class RMSProp(object):
    def __init__(self, rho):
        self.rho = rho
        self.s = .0

    def __call__(self, fun, x, step):
        g = fun.derivative(x, step)
        g2 = np.multiply(g, g)

        self.s = self.rho * self.s + (1 - self.rho) * g2

        return g / (np.sqrt(self.s) + 1e-8)


class Adam(object):
    def __init__(self, beta1, beta2):
        self.beta1 = beta1
        self.beta2 = beta2

        self.m = self.v = .0

    def __call__(self, fun, x, step):
        g = fun.derivative(x, step)
        g2 = np.multiply(g, g)

        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g2

        m_hat = self.m / (1 - self.beta1)
        v_hat = np.sqrt(self.v / (1 - self.beta2))
        
        return m_hat / (v_hat + 1e-8)


def compare_optimizers():
    solver = Solver(learning_rate = 1e-3, accuracy = 1e-8)
    
    for i in range(10):
        print(f'run {i}')
        fun = Function(noise_size = 0.5, circular = False)
        solution = fun.center
        
        sgd = SGD()
        rmsp = RMSProp(rho = 0.9)
        adam = Adam(beta1 = 0.9, beta2 = 0.999)

        res1, step1 = solver(fun, sgd)
        res2, step2 = solver(fun, rmsp)
        res3, step3 = solver(fun, adam)
        
        error1 = math.sqrt(np.dot(res1 - solution, res1 - solution))
        error2 = math.sqrt(np.dot(res2 - solution, res2 - solution))
        error3 = math.sqrt(np.dot(res3 - solution, res3 - solution))

        print(f'solution={solution}')
        print(f'res1={res1}, error1={error1}, step1={step1}')
        print(f'res2={res2}, error2={error2}, step2={step2}')
        print(f'res3={res3}, error3={error3}, step3={step3}')
        print()
    
if __name__ == "__main__":
    compare_optimizers()
