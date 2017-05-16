#!/usr/bin/python3

"""
Statistics and probability
"""

import sys
import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import binom

def main(args):
    s = sp.randn(5)

    # descriptive stats
    # print("Mean : {0:8.6f}".format(s.mean()))
    # print("Minimum : {0:8.6f}".format(s.min()))
    # print("Maximum : {0:8.6f}".format(s.max()))
    # print("Variance : {0:8.6f}".format(s.var()))
    # print("Std. deviation : {0:8.6f}".format(s.std()))
    # print(stats.describe(s))

    # uniform dist
    print("=== Uniform dist. ===")
    rv = stats.uniform()
    x = np.linspace(0, 1, 10)
    print(rv.pdf(x))

    # binom dist.
    print("=== Binom dist. ===")
    rv = stats.binom(10, .25)
    for i, x in enumerate(range(0, 10)):
        print(i, rv.pmf(x))

    # Poisson dist.
    print("=== Poisson dist. ===")
    rv = stats.poisson(1.5*4)
    for i, x in enumerate(range(0, 10)):
        print(i, rv.pmf(i))


if __name__ == "__main__":
    main(sys.argv[1:])