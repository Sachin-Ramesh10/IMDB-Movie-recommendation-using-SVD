import pandas as pd
from numpy.linalg import *
import numpy as np
from user_item_CF import user_based_cf,item_based_cf


def cf_svd():
    df = pd.read_csv('ratings.csv', sep=',', skiprows=1, header=None)
    df.columns = ['User', 'Movie', 'rating']
    matrix = df.pivot(index='User', columns='Movie', values='rating')
    matrix = matrix.fillna(value=-1)

    print("created matrix")

    U, s, V = linalg.svd(matrix)

    S = np.diag(s)

    row, column = U.shape
    rowv, columnv = V.shape
    sr, sc = S.shape

    Fu = np.delete(U, range(2, row + 1), 1)
    Fv = np.delete(V, range(2, columnv + 1), 0)
    Sm = np.delete(S, range(2, row + 1), 1)
    Fs = np.delete(Sm, range(2, sc + 1), 0)

    midp = Fu.dot(Fs)
    Xk = midp.dot(Fv)

    XK = pd.DataFrame(Xk)
    print(XK)

    print("Entering IB CF")
    ib_pred = item_based_cf(18, 458, 5, XK)
    print("Prediction by Item Based CF =" + "\t" + str(ib_pred))

    print("Entering UB CF")
    ub_pred = user_based_cf(18, 458, 5, XK)
    print("Prediction by User Based CF =" + "\t" + str(ub_pred))

cf_svd()


