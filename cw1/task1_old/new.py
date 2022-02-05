"""trying from the start because the rest is 
making my head hurt"""
"""
#Trying to get diagonals included in search
#################################################
comments :09/01/22
"""


#%%
#import statements 
import numpy as np
#%%
#defining functin

def pre_built(vol_bi_img):
    """this is the in built distance transform to compare with"""
    return ndimage.distance_transform_edt(vol_bi_img)

#%%
#setting arrays
easy = np.array([
    [0,0,1,1,0],
    [0,1,1,0,1],
    [1,1,1,1,0],
    [0,1,1,1,0],
    [0,0,1,0,0]])

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


#%%
euc_dist = np.zeros(shape=easy.shape)
sidex = [-1,1,0,0]
sidey = [0,0,-1,1]
diagx = [-1,-1,1,1]
diagy = [-1,-1,1,1]
#%%
for i in range(3): 
            adjx = x + newx[i]
            adjy = y + newy[i]
#%%
print(easy)
#indexes of values for which no distance yet
ind= []
m = easy.shape[0]
n = easy.shape[1]
for i in range(m):
    for j in range(n):
        if easy[i,j] == 0:
            #if value is 0, distance from itself is 0
            euc_dist[i,j] = 0 
        else:
            ind.append([i, j])

poped = []
while (len(ind)):
    poped = ind[0]
    ind.pop(0)

    x = poped[0]
    y = poped[1]
    #print("x,y" + str((x,y)), end = ' ')
    # now check for all adjancent of popped element
    for i in range(4):
            adjx = x + sidex[i]
            adjy = y + sidey[i]
            print(adjx,adjy)
            # make sure new coordinates, within array boundary
            #
            if (adjx >= 0 and adjx < m and adjy >= 0 and
                adjy < n and dist[adjx][adjy] > dist[x][y] + 1):
                #print("adjx,adjy" + str((adjx,adjy)), end = ' ')
                #print("x,y" + str((x,y)), end = ' ') 
                #adj -> positions of 0s
                #x,y -> coorsponding 1
                # update distance
                dist[adjx][adjy] = round((dist[x][y] + 1), ndigits=3)
                q.append([adjx, adjy])
                #euc[adjx][adjy] = np.sqrt(((x-adjx)**2)+((y-adjy)**2))
                euc_dist[x][y] = np.sqrt(dist)        
#%%            
elif:
    for each in range[sidex]: easy == 0:
    euc_dist[i,j] = 4
          #      print each
#            while easy[newx,newy] != 0:

 #       euc_dist[i,j] = easy[1,j]*+4

        
print (euc_dist)

# %%
#working ish code
def nearestOne(mat):
    """working, but missing diagonal neighbours"""
    #mat = mdup_data
    m = mat.shape[0]
    n = mat.shape[1]
    dist = np.zeros(shape = mat.shape)

    # two array when respective values of newx and
    # newy are added to (i,j) it gives up, down,
    # left or right adjacent of (i,j) cell
    newx = [-1, 0, 1, 0]
    newy = [0, -1, 0, 1]
        
    # queue of pairs to store nodes for bfs
    q = []
        
    # traverse matrix and make pair of indices of
    # cell (i,j) having value '1' and push them
    # in queue
    for i in range(m):
        for j in range(n):
            dist[i][j] = INT_MAX
            if (mat[i][j] == 0):
                    
                # distance of '0' from itself is always 0
                dist[i][j] = 0
                    
                # make pair and push it in queue
                q.append([i, j])
        
    # now do bfs traversal
    # pop element from queue one by one until it gets empty
    # pair element to hold the currently popped element
    poped = []
    while (len(q)):
        poped = q[0]
        q.pop(0)
            
        # coordinate of currently popped node
        # position of 0
        x = poped[0]
        y = poped[1]
        #print("x,y" + str((x,y)), end = ' ')
        # now check for all adjancent of popped element
        for i in range(4):
                
            adjx = x + newx[i]
            adjy = y + newy[i]
                
            # if new coordinates are within boundary and
            # we can minimize the distance of adjacent
            # then update the distance of adjacent in
            # distance matrix and push this adjacent
            # element in queue for further bfs
            if (adjx >= 0 and adjx < m and adjy >= 0 and
                adjy < n and dist[adjx][adjy] > dist[x][y] + 1):
                #print("adjx,adjy" + str((adjx,adjy)), end = ' ')
                #print("x,y" + str((x,y)), end = ' ') 
                #adj -> positions of 0s
                #x,y -> coorsponding 1
                # update distance
                dist[adjx][adjy] = dist[x][y] + 1
                q.append([adjx, adjy])
                #euc[adjx][adjy] = np.sqrt(((x-adjx)**2)+((y-adjy)**2))

    return np.sqrt(dist)

def nearestOne3d(mat):
    """3d one working, but not super accurate
    will try to integrate diagonal neighbours
    """
#%%
    mat = easy3d
    m = mat.shape[0]
    n = mat.shape[1]
    l = mat.shape[2]
    dist = np.zeros(shape = mat.shape)
    euc = np.zeros(shape = mat.shape)
    # up,down,left,right,backwards, forwards
    newx = [-1, 1, 0, 0,0,0]
    newy = [0, 0, -1, 1,0,0]
    newz = [0,0,0,0,-1,1]    
    # queue 
    q = []
    # traverse matrix and make pair of indices of
    # cell (i,j,k) having value '0' and push them
    # in queue
    for i in range(m):
        for j in range(n):
            for k in range(l):
                dist[i][j][k] = INT_MAX
                euc[i][j][k] = 255
                #print(mat[i][j][k])
                if (mat[i][j][k] == 0):
                    #print(i,j,k)
                    # distance of '0' from itself is always 0
                    dist[i][j][k] = 0
                    euc[i][j][k] = 0
                    # make pair and push it in queue
                    q.append([i, j, k])
    
    # pop element from queue one by one until it gets empty
    # pair element to hold the currently popped element
    poped = []
    while (len(q)):
        poped = q[0]
        q.pop(0)
            
        # coordinate of currently popped node
        # position of 0
        x = poped[0]
        y = poped[1]
        z = poped[2]

        #print("x,y" + str((x,y)), end = ' ')
        # now check for all adjancent of popped element
        for i in range(5):
            adjx = x + newx[i]
            adjy = y + newy[i]
            adjz = z + newz[i]
            # if new coordinates are within boundary and
            # we can minimize the distance of adjacent
            # then update the distance of adjacent in
            # distance matrix and push this adjacent
            # element in queue for further bfs
            if (adjx >= 0 and adjx < m and 
                adjy >= 0 and adjy < n and 
                adjz >= 0 and adjz < l and
                dist[adjx][adjy][adjz] > dist[x][y][z] + 1):
                #adj -> positions of 0s
                #x,y -> coorsponding 1
                # update distance
                dist[adjx][adjy][adjz] = dist[x][y][z] + 1
                q.append([adjx, adjy,adjz])
            
            #somehow still doesn't work 
            #euc[adjx][adjy][adjz] = np.sqrt((((adjx+1)-x)**2)+(((adjy+1)-y)**2)+(((adjz+1)-z)**2))
    print(euc[0])            
    print(np.sqrt(dist)[0])
    print(mat[0])
    #return np.sqrt(dist)

# %%

#print(easy)
#print(nearestOne(easy))
#print("___________________")
print(mdup_data)
a1 = nearestOne(mdup_data)
a2 = pre_built(mdup_data)

print(np.array_equal (a1, a2, equal_nan=False))
print(np.allclose(a1,a2, rtol=.2, atol=.2))

#%%
#print(pre_built(mdup_data))
#print(mdup_data)
for i in [0,1,2]:
    plt.subplot(1,2,1)
    plt.imshow(nearestOne3d(easy3d)[i,:,:])
    plt.subplot(1,2,2)
    plt.imshow(pre_built(easy3d)[i,:,:])
    plt.show()

for i in [0,1,2]:
    plt.subplot(1,2,1)
    plt.imshow(nearestOne(easy3d[i,:,:]))
    plt.subplot(1,2,2)
    plt.imshow(pre_built(easy3d[i,:,:]))
    plt.show()
# %%
