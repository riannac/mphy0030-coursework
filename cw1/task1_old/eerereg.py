# Python3 program to find distance of nearest
# cell having 1 in a binary matrix.
#%%
from collections import deque

MAX = 500
N = 3
M = 4

# Making a class of graph with bfs function.
g = [[] for i in range(MAX)]
n, m = 0, 0

# Function to create graph with N*M nodes
# considering each cell as a node and each
# boundary as an edge.
def createGraph():
    
    global g, n, m
    
    # A number to be assigned to a cell
    k = 1  

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            
            # If last row, then add edge on right side.
            if (i == n):
                
                # If not bottom right cell.
                if (j != m):
                    g[k].append(k + 1)
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

# BFS function to find minimum distance
def bfs(visit, dist, q):
    
    global g
    while (len(q) > 0):
        temp = q.popleft()
        
        for i in g[temp]:
            if (visit[i] != 1):
                print(i, temp)
                print(dist[i],dist[temp]+1)
                dist[i] = min(dist[i], dist[temp] + 1)
                q.append(i)
                visit[i] = 1
    print(dist)
    return dist

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
    
    global g, n, m
    
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
        dist[i] = 10**9
        visit[i] = 0

    # Inserting nodes whose value in matrix
    # is 1 in the queue.
    k = 1
    q =  deque()
    for i in range(N):
        for j in range(M):
            if (mat[i][j] == 1):
                dist[k] = 0
                visit[k] = 1
                q.append(k)
                
            k += 1

    # Calling for Bfs with given Queue.
    
    dist = bfs(visit, dist, q)
    
    # Printing the solution.
    prt(dist)

# Driver code
if __name__ == '__main__':
    
    mat = [ [ 0, 0, 0, 1 ],
            [ 0, 0, 1, 1 ],
            [ 0, 1, 1, 0 ] ]

    findMinDistance(mat)


# %%
