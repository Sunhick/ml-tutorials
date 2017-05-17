#!/usr/bin/python3

"""
probability distributions.

"""

import sys

import numpy as np
import scipy as sp
from scipy.stats import binom

def main(args):
    n = 17
    dist = binom(n, .765)

    print("Mean =", dist.mean())
    print("Var =", dist.var())
    print("Std =", dist.std())

    x = np.array([0,1,2,3,4,5,6])
    print("prob dist =", dist.pmf(x))

    samples = dist.rvs(size = 10000000)
    print(set(samples), n)
    m = len(samples)
    p_est = sum(samples)/(m*n)
    print("p(est) = {0} n = {1}".format(p_est, n))

if __name__ == "__main__":
    main(sys.argv[1:])