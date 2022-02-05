#%%
import numpy as np
import urllib.request
from numpy.core.numeric import allclose
from scipy import ndimage
import time
import matplotlib.pyplot as plt
def pre_built(vol_bi_img):
    """this is the scipy distance transform to compare with"""
    return ndimage.distance_transform_edt(vol_bi_img)


easy = np.array([
    [1,0,0,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1]])

mdup_data = np.array([
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,0,0,0,1,1,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,0,0,0,0,1,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,0,0,0,1,1,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 1, 1, 1, 1,0,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 1, 1, 1,1,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 1, 1,1,1,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,1,1,1,1,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 ])

easy3d = np.array([
        [[ 0, 1, 1, 1, 0],
        [ 1, 1, 1, 1, 1],
        [ 0, 1, 1, 1, 0],
        [ 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]],

       [[ 0, 0, 1,1,0],
        [0, 0, 1,1,1],
        [1, 0, 1,1,0],
        [0,0,0,1,1],
        [0,0,0,0,1]],

       [[0, 0, 0,1,1],
        [0, 0, 1,1,1],
        [1, 0, 0,1,0],
        [0,0,0,1,1],
        [0,0,1,1,1]]])




# Python3 program to find distance of nearest
# cell having 1 in a binary matrix.
from collections import deque
mat = easy
MAX = 500
N = mat.shape[0]
M = mat.shape[1]
n =N
m=M
# Making a class of graph with bfs function.
g = [[] for i in range(MAX)]
gg = np.zeros(shape = (mat.shape))

#index value arrays
# padded to include external boundary

gx = (np.arange(-1,n+1)*np.ones(shape =(n+2,m+2)))
gy = (np.reshape(np.arange(-1,n+1),(m+2,1)))*np.ones(shape=(n+2,m+2))
#%%

for n in range(N+2):
   print()
   for m in range(M+2):
        print((gx[n][m],gy[n][m]), end= ' ')   


#%%

# BFS function to find minimum distance
def bfs(visit, dist, q):
    
    while (len(q) > 0):
        temp = q.popleft()

        for i in g[temp]:
            if (visit[i] != 1):
                dist[i] = min(dist[i], dist[temp] + 1)
                q.append(i)
                visit[i] = 1
                
    return dist

#%%

# Printing the solution.
def prt(dist):
    
    c = 1
    for i in range(1, n * m + 1):
        print(dist[i], end = " ")
        if (c % m == 0):
            print()
            
        c += 1

# Find minimum distance
def findMinDistance(mat):
#%%
    mat = easy
    MAX = 500
    N = mat.shape[0]
    M = mat.shape[1]
    n =N
    m=M
    
    
    # Creating a graph with nodes values assigned
    # from 1 to N x M and matrix adjacent.
    gx = (np.arange(-1,n+1)*np.ones(shape =(n+2,m+2)))
    gy = (np.reshape(np.arange(-1,n+1),(m+2,1)))*np.ones(shape=(n+2,m+2))

    # To store minimum distance
    e_dist = np.zeros(shape=mat.shape)
    e_dist = 9999*e_dist
    dist = [0] * MAX
    
    #print(dist)
    # To mark each node as visited or not in BFS
    visit = [0] * MAX
    e_visit = np.zeros(shape=mat.shape)
    e_visit = 9999*e_visit

    
#print(visit) 

    # Initialising the value of distance and visit.
    for i in range(n):
        for j in range(m):
            e_dist[i][j] = 999
            e_visit[i][j] = 0
    print(mat)
    # Inserting nodes whose value in matrix
    # is 0 in the queue.
    k = 1
    q =  deque()
    qx = 1
    xq = []
    xyq= []
    qy = 1
    yq=[]
    for i in range(N):
        for j in range(M):
            if (mat[i][j] == 0):
                
                #e = ((i-x)**2+(i-j)**2)
                e_dist[i][j] = 0
                e_visit[i][j] = 1
                q.append(k)
                xyq.append((i,j))
                #print(k)
                
            k += 1
            #qy += 1
            #qx += 1
    
    print(xyq)
#    print(yq)
    #print(gx)
    #print(e_dist)
    #print(e_visit)
    # Calling for Bfs with given Queue.
    dist = bfs(visit, dist, q)
    #print(q)

    # Printing the solution.
    #prt(dist)
#%%
# Driver code
if __name__ == '__main__':
    mat = easy
    #mat = [ [ 0, 0, 0, 1 ],
     #       [ 0, 0, 1, 1 ],
      #      [ 0, 1, 1, 0 ] ]

    findMinDistance(mat)

# This code is contributed by mohit kumar 29
# %%
