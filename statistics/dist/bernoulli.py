#!/usr/bin/python3

"""
Bernoulli distributions

bernoulli.pmf(k) = 1-p  if k = 0
                 = p    if k = 1

"""

import sys

import numpy as np
import scipy as sp
from scipy.stats import bernoulli


def main(args):
    p = .36
    dist = bernoulli(p)

    r = np.array([0,1,1,0])
    print("prob dist =", dist.pmf(r))
    print("Mean =", dist.mean())
    print("Var =", dist.var())
    print("Std =", dist.std())
    
    n = 10000000
    # generate n random numbers using bernoulli dist.
    # r = bernoulli.rvs(p, size=1000)
    sample = dist.rvs(size = n)

    # since all numbers in r are either 0 or 1. just sum them to count the 
    # number of ones.
    pe = sum(sample)
    print("p(estimate) = {0} p(actual) = {1}".format(pe/len(sample), p))

    # fit a bernoulli dist.
    sample = np.random.randint(2, size = 1000)
    pe = sum(sample)/len(sample)
    print("p(est) =", pe)


if __name__ == "__main__":
    main(sys.argv[1:])
