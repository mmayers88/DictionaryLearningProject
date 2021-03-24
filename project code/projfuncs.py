#=====================================================================================
# Authors:          Andrew, Mikhail, Anthony
# Last Updated:     Feb 19, 2021
#
# Notes:            MSP Algorithm from [1] As  interpreted by team.
#=====================================================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds, eigs
from tqdm import tqdm
import itertools

import random
'''
Used for dictionary initialization
Inputs:
  size of matrix
Returns:
  Orthogonal random matrix
'''
def radU(N): #call it radU for coolness
  X = np.random.randn(N,N)
  [Q,R] = np.linalg.qr(X)
  r = np.sign(np.diag(R))
  return Q * r.T

'''
l4_max is the MSP algorithm Described in paper
Inputs:
  Y is the observed data 
  max_iter is the number of iterations
Returns:
  At is the approximated Learned dictionary 
'''
def l4_max(Y, max_iter):
  M, N = Y.shape
  At = radU(M)
  for kk in tqdm(range(max_iter)):
    Aprev = At
    AtY = Aprev @ Y
    dA = (AtY**3) @ Y.T
    U, S, VT = np.linalg.svd(dA, compute_uv=True)
    At = U@VT
  return At 