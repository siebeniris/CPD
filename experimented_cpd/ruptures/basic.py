import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np


# generate signal
n = 500  # number of samples
dim = 2  # number of dimension
n_bkp, sigma = 3, 5  # number of change points, noise standard deviation
signal, bkps = rpt.pw_constant(n, dim, n_bkp, noise_std=sigma)


