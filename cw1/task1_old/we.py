# Python3 program to find distance of nearest
# cell having 1 in a binary matrix.
#%%
import numpy as np
import urllib.request
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
#%%
def mine(mat):  
#%%   
    mat =easy
    MAX = 500
    N = mat.shape[0]
    M = mat.shape[1]

    # Making a class of graph with bfs function.
    g = [[] for i in range(MAX)]
    #g = np.zeros(shape = mat.shape)
    n, m = 0, 0
    n = N
    m = M
    # Function to create graph with N*M nodes
    # considering each cell as a node and each
    # boundary as an edge.
    def createGraph():
#%%
        # A number to be assigned to a cell
        n = N
        m = M
        k = 1  

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                
                # If last row, then add edge on right side.
                if (i == n):
                    
                    # If not bottom right cell.
                    if (j != m):
                        g[k].append(k + 1)
                        #print(k)
                        g[k + 1].append(k)

                # If last column, then add edge toward down.
                elif (j == m):
                    g[k].append(k+m)
                    g[k + m].append(k)
                    
                # Else makes an edge in all four directions.
                else:
                    g[k].append(k + 1)
                    g[k + 1].append(k)
                    g[k].append(k+m)
                    g[k + m].append(k)

                k += 1
    #n, m = N, M
    #print(createGraph())
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

    # Printing the solution.
    def prt(dist):
        w = np.zeros(shape = (n,m))
        c = 1 
        a = 1
        b=1
        for i in range(1, n * m + 1):
            w[b-1][a-1] = (dist[i])
            if (c % m == 0):
                b += 1
                a = 0
            c+=1
            a+=1
        return w
        #print(np.sqrt(w))
        #return(np.sqrt(w))

    # Find minimum distance
    def findMinDistance(mat):
        
        
        
        # Creating a graph with nodes values assigned
        # from 1 to N x M and matrix adjacent.
        n, m = N, M
        createGraph()
        # To store minimum distance
        dist = [0] * MAX

        # To mark each node as visited or not in BFS
        visit = [0] * MAX
        
        # Initialising the value of distance and visit.
        for i in range(1, M * N + 1):
            dist[i] = n*m
            visit[i] = 0

        # Inserting nodes whose value in matrix
        # is 1 in the queue.
        k = 1
        q =  deque()
        for i in range(N):
            for j in range(M):
                if (mat[i][j] == 0):
                    dist[k] = 0
                    visit[k] = 1
                    q.append(k)
                    
                k += 1

        # Calling for Bfs with given Queue.
        dist = bfs(visit, dist, q)
        
        
        return prt(dist)
        # Printing the solution.
        #prt(dist)

    # Driver code
    if __name__ == '__main__':
        
        mat = mdup_data
        #mat = [ [ 0, 0, 0, 1 ],
        #       [ 0, 0, 1, 1 ],
        #      [ 0, 1, 1, 0 ] ]

        return findMinDistance(mat)

print("hello")

a = easy
print(mine(a))
print(pre_built(a)**2)


#a1 =(mine(a))
#print(a1)
#print(a2**2)
#%%
"""
print(np.array_equal (a1, a2, equal_nan=False))
print(np.allclose(a1,a2, rtol=.2, atol=.2))
my_alg_time = time.time()

a1 =(mine(a))
print("--- %s seconds ---" % (time.time() - my_alg_time))

pre_alg_time = time.time()
a2 = pre_built(a)
print("--- %s seconds ---" % (time.time() - pre_alg_time))

# values
print(np.array_equal (a1, a2, equal_nan=False))
print(np.allclose(a1,a2, rtol=.2, atol=.2))

#ploting
plt.figure()
plt.subplot(1,2,1)
plt.imshow(a1)
plt.subplot(1,2,2)
plt.imshow(a2)
plt.show()
# This code is contributed by mohit kumar 29
"""
# %%
