import numpy as np
from scipy.spatial.distance import cosine


def user_based_cf(userId, movieId , neighbor, mat):

    result = {}
    kmean = []
    total = 0.0
    a = userId
    k = neighbor
    movid = movieId

    umean = mat.mean(axis=1)

    other = mat.loc[:, movid].values

    other[np.isnan(other)] = 0

    mmat = mat
    print("checking for null")

    del mmat[movid]
    print("cleared null")

    pred_rat = umean.get_value(a)

    userrat = mmat.loc[[a]].values
    userrat[np.isnan(userrat)] = 0
    s1 = list(mmat.index)

    print("creating sim")

    for i in s1:

        de1 = mmat.loc[[i]].values
        de1[np.isnan(de1)] = 0
        result[a, i] = 1 - cosine((userrat), (de1))

    del result[a, a]
    print("sorting")
    sim = sorted(result, key=result.get, reverse=True)
    print("created similarity")

    for i in range(0, k):
        c, b = sim[i]
        kmean.append(b)
        total = total + result[sim[i]]

    print("calculating prediction")
    for i in range(0, k):
        d, b = sim[i]
        pred_rat = pred_rat + ((result[sim[i]] * (other[b - 1] - umean.get_value(kmean[i]))) / total)

    return pred_rat

def item_based_cf(userId, movieId , neighbor, mat1):


    result = {}
    kmean = []
    total = 0.0
    a = userId
    k = neighbor
    movid = movieId
    mmat1 = ""


    umean = mat1.mean(axis=0)
    other = mat1.loc[a].values
    other[np.isnan(other)] = 0

    pred_rat = umean.get_value(movid)

    mmat1 = mat1.drop(a)
    print("dropped column")


    userrat = mmat1.loc[:, movid].values
    userrat[np.isnan(userrat)] = 0
    print("calculating sim")

    s = list(mmat1.columns.values)
    for i in s:

        de = mmat1.loc[:, i].values
        de[np.isnan(de)] = 0
        result[movid, i] = 1 - cosine((de), (userrat))
    del result[movid, movid]

    print("calculated similarities")
    sim = sorted(result, key=result.get, reverse=True)
    for i in range(0, k):
        c, b = sim[i]
        kmean.append(b)
        total = total + result[sim[i]]

    for i in range(0, k):
        d, b = sim[i]
        pred_rat = pred_rat + ((result[sim[i]] * (other[b - 1] - umean.get_value(kmean[i]))) / total)

    return pred_rat
