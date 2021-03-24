import scipy
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import argparse



def encode_mkr(D_dictionary,img):
    x = lstsq(D_dictionary, img)
    x1 = x[0]
    x1 = x1.reshape(784,1)
    return x1

def reconstruct(D_dictionary,x1):
    testo = D_dictionary @ x1
    testo = testo.reshape(28,28)
    plt.imshow(testo)
    plt.show()
    return testo

def test(dName = 'dic.npy',bName = 'number.npy',vec = False):
    dictionary = np.load(dName)
    base = np.load(bName)
    if vec == False: #this could always just be done by detecting shape
        base = base.reshape(784,1)
    plt.imshow(base.reshape(28,28))
    plt.show()
    print("Encoding...")
    x = encode_mkr(dictionary,base)
    reconstruct(dictionary,x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dicName', '-d', default='dic.npy', type=str,
                        help='numpy dictionary filename')
    parser.add_argument('--baseName', '-b', default='number.npy', type=str,
                        help='numpy noisy number filename')
    parser.add_argument('--vec', '-v', default=False, type=bool,
                        help='default false for matrix')


    args = parser.parse_args()

    test(args.dicName,args.baseName,args.vec)
