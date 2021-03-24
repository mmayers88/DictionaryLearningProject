#=====================================================================================
# Authors:          Andrew, Mikhail, Anthony
# Last Updated:     Feb 19, 2021
#
# Notes:            Used for general tests and to show created atoms.
#                   Best bases extracted using code from [5]
#=====================================================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import svds, eigs
from tqdm import tqdm
import itertools
import scipy

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
"""#MNISR Data Load"""

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "sample_data/"
train_data = np.loadtxt(data_path + "mnist_train_small.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

img = train_imgs[0].reshape((28,28))
img = test_imgs[0].reshape((28,28))
print("Label: ",test_labels[0])
#plt.imshow(img, cmap="Greys")

X = np.array([[1,2,3],[4,5,6],[7,8,9]])

max_iter = 10
train = train_imgs

## Clean learning ##
D_patches = train_imgs

D_patches = D_patches.T

D_dictionary = l4_max(D_patches, max_iter) #actual learning being done
D_dictionary = D_dictionary.T
COLOR_SCALE = {
    'vmin': 0,
    'vmax': 255
}
D_sparse_encoding = D_dictionary.T @ D_patches
norms = np.abs(D_sparse_encoding)
norms = np.sum(norms, axis=1)
all_indices = list(range(len(norms)))
all_indices.sort(key=lambda row: norms[row], reverse=True)
sum_signs = np.sum(D_sparse_encoding, axis=1)
sum_signs = np.sign(sum_signs)
ROWS, COLS = 3, 4
fig, axs = plt.subplots(ROWS, COLS, figsize=(10, 8))
plt.subplots_adjust(left=None, right=None, bottom=None, top=None, wspace=0.05, hspace=0.05)
for index, ax in zip(all_indices, axs.flat):
    base = D_dictionary[:,index] * sum_signs[index]
    base = base - base.min()
    base = base / base.max() * 255
    base = np.reshape(base, (28,28))
    ax.imshow(base, **COLOR_SCALE, cmap='gray')
    ax.axis('off')
plt.show()

## Noisy learning ##
D_patches = train_imgs + np.random.normal(scale=.3333, size=train_imgs[0].shape)


D_patches = D_patches.T


D_dictionary = l4_max(D_patches, max_iter) #actual learning being done
D_dictionary = D_dictionary.T
COLOR_SCALE = {
    'vmin': 0,
    'vmax': 255
}
D_sparse_encoding = D_dictionary.T @ D_patches
norms = np.abs(D_sparse_encoding)
norms = np.sum(norms, axis=1)
all_indices = list(range(len(norms)))
all_indices.sort(key=lambda row: norms[row], reverse=True)
sum_signs = np.sum(D_sparse_encoding, axis=1)
sum_signs = np.sign(sum_signs)
ROWS, COLS = 3, 4
fig, axs = plt.subplots(ROWS, COLS, figsize=(10, 8))
plt.subplots_adjust(left=None, right=None, bottom=None, top=None, wspace=0.05, hspace=0.05)
for index, ax in zip(all_indices, axs.flat):
    base = D_dictionary[:,index] * sum_signs[index]
    base = base - base.min()
    base = base / base.max() * 255
    base = np.reshape(base, (28,28))
    ax.imshow(base, **COLOR_SCALE, cmap='gray')
    ax.axis('off')
plt.show()

np.save('dic.npy',D_dictionary)
dirty =  train_imgs[1] + np.random.normal(scale=.3333, size=train_imgs[0].shape)
np.save('number.npy',dirty.reshape(28,28))

x = scipy.sparse.linalg.lsqr(D_dictionary, train_imgs[1].T)
x1 = x[0]
x1 = x1.reshape(784,1)
testo = D_dictionary @ x1
testo = testo.reshape(28,28)
plt.imshow(testo)
plt.show()