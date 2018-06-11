import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
import csv

result = []
movieID = {}
userId = {}

movies = ["Shawshank Redemption, The (1994)","Godfather, The (1972)","Godfather: Part II, The (1974)","Dark Knight, The (2008)",
          "12 Angry Men (1957)","Schindler's List (1993)","Pulp Fiction (1994)","Lord of the Rings: The Return of the King, The (2003)",
          "Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966)","Fight Club (1999)"]

with open('movies.csv', encoding= 'UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter= ',')
    for line in reader:
        if line[1] in movies:
            movieID[line[1]] = line[0]
            result.append(line[0])

with open('ratings.csv', encoding= 'UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter= ',')
    for line in reader:
        if line[1] in result:
            userId[line[1]] = line[0]
            with open('IMDB.csv', 'a', newline='') as f:
                wtr = csv.writer(f, delimiter= ',')
                wtr.writerows([line])

df = pd.read_csv('IMDB.csv', sep=',', header=None)
df.columns = ['User', 'Movie', 'rating']
matrix = df.pivot(index='User', columns='Movie', values='rating')
matrix = matrix.fillna(value=-1)

print("doing svd")

U, s, V = linalg.svd(matrix)

print("done")

Um = pd.DataFrame(data=U[0:,0:], index=U[0:,0],columns=U[0,0:])
Vm = pd.DataFrame(data=V[0:,0:], index=V[0:,0],columns=V[0,0:])
U1 = Um.iloc[:,0].values
U2 = Um.iloc[:,1].values
M1 = Vm.iloc[0].values
M2 = Vm.iloc[1].values

plt.plot(U1, U2, 'ro',M1,M2 ,'go')
plt.axis([-0.8, -0.2, -1.2, 0.6])
plt.show()
