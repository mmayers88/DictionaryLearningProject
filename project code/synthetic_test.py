#=====================================================================================
# Authors:          Andrew, Mikhail, Anthony
# Last Updated:     Feb 19, 2021
#
# Notes:            Used for synthetic tests and plots.
#                   Used to test changes of features and number of samples.
#=====================================================================================
import generator as g
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import os
from projfuncs import l4_max, radU
import seaborn as sns
from tqdm import tqdm


features = 50
s = 20000
t = 0.3
# samples = np.linspace(500,10000,20, dtype=int)
# theta = np.linspace(0,1,20)
# htmp1 = np.zeros((20,20))
# htmp2 = htmp1.copy()
err1 = []
err2 = []
max_iter = 60
# for i,t in enumerate(theta):
#     for j,s in enumerate(samples):

Y, D, X  = g.random_dictionary_learning_instance(features, s, t)

# my_D = l4_max(Y, max_iter)
#l4_max with other error
M, N = Y.shape
At = radU(M)
for kk in tqdm(range(max_iter)):
    Aprev = At
    AtY = Aprev @ Y
    dA = (AtY ** 3) @ Y.T
    U, S, VT = np.linalg.svd(dA, compute_uv=True)
    At = U @ VT
    err1.append(g.sum_of_fourth_powers(At @ Y) / (3 * features * s * t))
    err2.append(g.sum_of_fourth_powers(At @ D)/features)
# my_D = my_D.T
# print(abs(1-g.sum_of_fourth_powers(my_D@D)/features))

# print(g.sum_of_fourth_powers(my_D @ Y) / (3 * features * s * t))
# htmp1[i,j] = abs(1-g.sum_of_fourth_powers(my_D@D)/features)
# htmp2[i,j] = g.sum_of_fourth_powers(my_D@Y)/(3*features*s*t)

# sparseX = my_D.T @ Y
# print(np.linalg.norm(Y-my_D @ (my_D.T @Y)))
# print(np.linalg.norm(Y-D@X))
# plt.imshow(htmp1)
# plt.show()
# plt.imshow(htmp2)
# plt.show()

plt.plot(err1, label=r'$||AY||_4^4/3np \theta$')
plt.plot(err2, label=r'$ ||AD_o||_4^4/n$')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Normalized Objective Value vs Iteration for n={}, p={}, $\\theta$={}'.format(features,s,t))
plt.legend()
plt.show()