import pandas as pd
from user_item_CF import user_based_cf,item_based_cf
import numpy as np

def get_predicion(userid, movieid, neighbor):

#removes time stamp column
    f = pd.read_csv("ratings.csv")
    keep_col = ['userId', 'movieId', 'rating']
    new_f = f[keep_col]
    new_f.to_csv("ratings.csv", index=False)
    print("removed timestamp")

#creates the user_Item matrix
    df = pd.read_csv('ratings.csv', sep=',', skiprows=1, header=None)
    df.columns = ['User', 'Movie', 'rating']
    matrix = df.pivot(index='User', columns='Movie', values='rating')
    matrix.set_value(userid, movieid, np.nan)
    print("created matrix for UBCF")
    ub_pred = user_based_cf(userid, movieid, neighbor, matrix)
    print("Prediction by User Based CF =" + "\t" + str(ub_pred))
	
#creates the user_Item matrix
    df = pd.read_csv('ratings.csv', sep=',', skiprows=1, header=None)
    df.columns = ['User', 'Movie', 'rating']
    mat2 = df.pivot(index='User', columns='Movie', values='rating')
    print("created matrix for IBCF")
	mat2.set_value(userid, movieid, np.nan)
    ib_pred = item_based_cf(userid, movieid, neighbor, mat2)
    print("Prediction by Item Based CF =" + "\t" + str(ib_pred))

#enter userid movieid and neighborhood size

get_predicion(18, 858, 5)