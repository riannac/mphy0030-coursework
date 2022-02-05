#%%
############################################################         
#           2d WORKING ! wrong ish 
#           3d WORKING ! wrong ish
#           not including diagonals  
"""

                       working on it 
"""
##########################################################

#works for 2d 
#%%
#import
import numpy as np
import urllib.request
from numpy.core.fromnumeric import sort
from numpy.core.numeric import indices
from numpy.lib.function_base import append
from numpy.lib.index_tricks import index_exp
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_bf
from skimage.io import imread,imsave
from matplotlib import pyplot as plt


#%%
#download
""" 
download label_train00.npy from download link provided in course work
“label_train00.npy” loaded as lbt_data contains a binary segmentation of Pelvic MR volume image
"""

lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)

#%% 
#function def
################################################################################
#                            Function Definitions                              #
################################################################################
def print_properties(x):
    print("\n type : " + str(type(x))
    + "\n size : " + str(x.size)
    + "\n shape : " + str(x.shape)
    + "\n dtype : " + str(x.dtype)
    + "\n ndim : " + str(x.ndim)
    + "\n itemsize : " + str(x.itemsize)
    + "\n nbytes : " + str(x.nbytes))
    

def distance_transform_np(vol_bi_img):
    """docstring to describe the algorithm used"""
    #input: 3D volumetric bianary image
    #output: 3d euclidean distance transform
    print("this will be the transform")
    print(vol_bi_img)

def pre_built(vol_bi_img):
    """this is the in built distance transform to compare with"""
    return ndimage.distance_transform_edt(vol_bi_img)
def brute(vol_bi_img):
    return ndimage.distance_transform_bf(vol_bi_img)

"""visualisatino while developing comparing image and it's transform and specified slices"""
def plot_comp(data, slice_index=15):
    """plot image and its transform to compare"""
    #not perfect 
    plt.figure()
    plt.subplot(1,2,1)
    if data.ndim ==3:
        plt.imshow(data[slice_index,:,:], cmap=plt.cm.gray)
        #commutes full transform but only plots 1 slice
        plt.imshow(pre_built(data)[slice_index,:,:])
    else:
        plt.imshow(data, cmap=plt.cm.gray)
        plt.imshow(pre_built(data))
   
    plt.subplot(1,2,2)
    plt.show()


def alg_comp(data, slice_index=15):
    """plot my transform and the transform to compare"""
    plt.figure()
    plt.subplot(1,2,1)
    if data.ndim ==3:
        data = data[slice_index, : ,:] 
    plt.imshow(nearestOne(data))
    plt.subplot(1,2,2)
    plt.imshow(pre_built(data))
    plt.show()

#%% 
#setting arrays
easy3d = np.array([
        [[ 0,  1,  1,1,0],
        [ 1,  1,  1, 1,1],
        [ 0,  1,  1,1,0]],

       [[ 0, 0, 1,1,0],
        [0, 0, 1,1,1],
        [1, 0, 1,1,0]],

       [[0, 0, 0,1,1],
        [0, 0, 1,1,1],
        [1, 0, 0,1,0]]])

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
easy = np.array([
    [0,0,1,1,0],
    [0,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,1],
    [0,0,1,0,0]])
#%%
#nearest 3d
MAX = 1000
INT_MAX = (2**32)
 
# distance matrix which stores the distance of
# nearest '1'


def nearestOne3d(mat):
    MAX = 1000
    INT_MAX = (2**32)
    m = mat.shape[0]
    n = mat.shape[1]
    l = mat.shape[2]
    dist = np.zeros(shape = mat.shape)
    #euc = np.zeros(shape = mat.shape)
    # two array when respective values of newx and
    # newy are added to (i,j) it gives up, down,
    # left or right adjacent of (i,j) cell
    newx = [-1, 0, 0, 1, 0, 0]
    newy = [0, -1, 0, 0, 1, 0]
    newz = [0, 0, -1, 0, 0 ,1]

    # (-1,0,0)
    # (0,-1,0)
    # (0,0,-1)
    # (1,0,0)
    # (0,1,0)
    # (0,0,1)
     
    # queue of pairs to store nodes for bfs
    q = []
     
    # traverse matrix and make pair of indices of
    # cell (i,j) having value '1' and push them
    # in queue
    for i in range(m):
        for j in range(n):
            for k in range(l):
                dist[i][j][k] = INT_MAX
                if (mat[i][j][k] == 0):
                    
                    # distance of '0' from itself is always 0
                    dist[i][j][k] = 0
                    
                    # make pair and push it in queue
                    q.append([i, j,k])
        
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
            if (adjx >= 0 and adjx < m and adjy >= 0 and
                adjy < n and dist[adjx][adjy][adjz] > dist[x][y][z] + 1):
                
                dist[adjx][adjy][adjz] = dist[x][y][z] + 1
                q.append([adjx, adjy, adjz])
            #else:
                    #euc[adjx][adjy] = np.sqrt(((x-adjx)**2)+((y-adjy)**2))
    
    return np.sqrt(dist)

#%%
#plotting things
plt.figure()
#plt.subplot(1,2,1)
plt.imshow
for i in [0,1,2]:
    plt.figure()
    plt.suptitle('2d compare')
    plt.subplot(1,3,1)
    plt.imshow(easy3d[i,:,:])
    plt.subplot(1,3,2)
    plt.imshow(nearestOne(easy3d[i,:,:]))
    plt.subplot(1,3,3)
    plt.imshow(pre_built(easy3d[i,:,:]))
#%%
for i in [0,1,2]:
    plt.figure()
    plt.suptitle('3d compare')
    plt.subplot(1,3,1)
    plt.imshow(easy3d[i,:,:])
    plt.subplot(1,3,2)
    plt.imshow(nearestOne3d(easy3d)[i,:,:])
    plt.subplot(1,3,3)
    plt.imshow(pre_built(easy3d)[i,:,:])
    plt.legend()
    plt.show()
#%%
for i in [0,1,2]:
    plt.subplot(1,2,1)
    plt.imshow(pre_built(easy3d[i,:,:]))
    plt.subplot(1,2,2)
    plt.imshow(pre_built(easy3d)[i,:,:])
    plt.show()


#%%
for n in [10,15,20,25]: 
    alg_comp(lbt_data,n)
#plot_comp(nearestOne(lbt_data[15,:,:]))


# %%
#dis_i_need
def dis_i_need(x,x1,y,y1):
    return np.sqrt((((x-x1)**2)+(y-y1)**2))
#%%
#print(dis_i_need(1,0,1,0))
def nearestOne(mat):

    #mat = easy
    MAX = 1000
    INT_MAX = (2**32)
    m = mat.shape[0]
    n = mat.shape[1]
    dist = np.zeros(shape = mat.shape)
    euc = np.zeros(shape = mat.shape)
    
    # two array when respective values of newx and
    # newy are added to (i,j) it gives up, down,
    # left or right adjacent of (i,j) cell
    newx = [-1, 0, 1, 0, -1,1,1,1]
    newy = [0, -1, 0, 1,-1,1,-1,1]
    #newyd= 
    

    # queue of pairs to store nodes for bfs
    q = []
     
    # traverse matrix and make pair of indices of
    # cell (i,j) having value '1' and push them
    # in queue
    for i in range(m):
        for j in range(n):
            dist[i][j] = INT_MAX
            euc[i][j] = INT_MAX
            
            if (mat[i][j] == 0):
                # distance of '0' from itself is always 0
                dist[i][j] = 0
                euc[i][j] = 0
            #else:
                q.append([i, j])
     
    
    # q= elements i don't have an answer for 
    poped = []
    while (len(q)):
        poped = q[0]
        q.pop(0)
         
        # coordinate of currently popped node
        # position of 1
        x = poped[0]
        y = poped[1]

        
        for i in range(len(newx)):
             
            adjx = x + newx[i]
            adjy = y + newy[i]
             

            if (adjx >= 0 and adjx < m and adjy >= 0 and
                adjy < n):
                if  dist[adjx][adjy] > dist[x][y] + 1:
                    #dist[adjx][adjy] > print(x,y)
                    #print(dis_i_need(x,adjx,y,adjy))
                    #euc[adjx][adjy] = dis_i_need(adjx,x,adjy,y)
                    #euc[adjx][adjy] = dis_i_need(x,adjx,y,adjy)

                    dist[adjx][adjy] = dist[x][y] + 1
                    q.append([adjx, adjy])
                    #print(adjx,adjy,x,y)
                    #np.sqrt(((x-adjx)**2)+((y-adjy)**2))
    return np.sqrt(dist)
    #print(euc)
mat = mdup_data
print(pre_built(mat))
print(nearestOne(mat))
    
#    return np.sqrt(dist)

# %%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(pre_built(mat))
plt.subplot(1,2,2)
plt.imshow(nearestOne(mat))
plt.show()


# %%
def nearesttry(mat):
    #trsh delete
#%%
    mat = easy
    MAX = 1000
    INT_MAX = (2**32)
    m = mat.shape[0]
    n = mat.shape[1]
    dist = np.zeros(shape = mat.shape)
    euc = np.zeros(shape = mat.shape)

    # two array when respective values of newx and
    # newy are added to (i,j) it gives up, down,
    # left or right adjacent of (i,j) cell
 

    # queue of pairs to store nodes for bfs
    q = []

    # traverse matrix and make pair of indices of
    # cell (i,j) having value '1' and push them
    # in queue
    for i in range(m):
        for j in range(n):
            dist[i][j] = 99
            euc[i][j] = 99
            
            if (mat[i][j] == 0):
                # distance of '0' from itself is always 0
                dist[i][j] = 0
                euc[i][j] = 0
            else:
                q.append([i, j])
     
    #print(euc)

    # q= elements i don't have an answer for 
    poped = []

    print(q)
    x_p = np.array([1, 1,-1,-1])
    y_p = np.array([1,-1, 1,-1])
    x_add = np.array([0, 1,-1,0])
    y_add = np.array([1,0, 0,-1])
#%%
q= 2
a = np.zeros(shape=(3+q,3+q))

w=1
#size of matrix = 
q = [0,1][0,0]
qx = [0,1,-1,0,1,-1,1,-1]
qy = [1,0,0,-1,1,-1,-1,1]


w=2
#size of matrix = 5x5
q = [0,2][1,2][2,2]
q =

w=3
#size of matrix = 7x7
q = [0,3][1,3][2,3][3,3]

print(a)

#%%
    

    for coo in q:
        x = coo[0]
        y = coo[1]
        a=1

        adjx = (x+1+a)
        adjy = (y+a)
        
        #print(adjx,adjy)
        for index in range(len(x_p)):
            ix = (x + a*x_add[index])
            iy = (y + a*y_add[index])
            
            if(ix >= 0 and ix < m and iy >= 0 and
                iy < n) and (mat[ix][iy] ==0):
                    euc[x,y] =dis_i_need(x,ix,y,iy)
            else:
                ix = 
        
    print(x,y)
    print(euc)
#%%
       
        if ((mat[perm[1]][adjy] ==0) or 
            (mat[adjx][adjy*(-1)] ==0) or
            (mat[adjx*(-1)][adjy] ==0) or
            (mat[adjx*(-1)][adjy*(-1)] ==0)):

        if ((mat[adjx][adjy] ==0) or 
            (mat[adjx][adjy*(-1)] ==0) or
            (mat[adjx*(-1)][adjy] ==0) or
            (mat[adjx*(-1)][adjy*(-1)] ==0)):


#%%
        print(mat[adjx][adjy], mat[adjx][adjy*(-1)],
            mat[adjx*(-1)][adjy],
            mat[adjx*(-1)][adjy*(-1)]) 
    print(mat)
#%%
        if(adjx >= 0 and adjx < m and adjy >= 0 and
                adjy < n):
            if ((mat[adjx][adjy] ==0) or 
            (mat[adjx][adjy*(-1)] ==0) or
            (mat[adjx*(-1)][adjy] ==0) or
            (mat[adjx*(-1)][adjy*(-1)] ==0)):
                euc[x][y] = dis_i_need(x,adjx,adjy,y)
        elif (adjxd >= 0 and adjxd < m and adjyd >= 0 and
            adjyd < n):
            if ((mat[adjxd][adjyd] ==0) or 
                (mat[adjxd][adjyd*(-1)] ==0) or
                (mat[adjxd*(-1)][adjyd] ==0) or
                (mat[adjxd*(-1)][adjyd*(-1)] ==0)):
                    euc[x][y] = dis_i_need(x,adjxd,adjyd,y)
        else: 
            a = a +1 
            #q.append[x,y]

    print("____________________")
    print(mat)
    print(euc)
#%%
    #while (len(q)):
    
        poped = q[0]
        q.pop(0)
        a = 0
        # coordinate of currently popped node
        # position of 1
        x = poped[0]
        y = poped[1]
        #print(x)
        adjx = (x+1+a)
        adjy = (y+a)

        adjxd = (x+1+a)
        adjyd = (y+1+a)
        #print(adjx)

        if(adjx >= 0 and adjx < m and adjy >= 0 and
                adjy < n):
            if ((mat[adjx][adjy] ==0) or 
            (mat[adjx][adjy*(-1)] ==0) or
            (mat[adjx*(-1)][adjy] ==0) or
            (mat[adjx*(-1)][adjy*(-1)] ==0)):
                euc[x][y] = dis_i_need(x,adjx,adjy,y)
            elif (adjxd >= 0 and adjxd < m and adjyd >= 0 and
                adjyd < n):
                if ((mat[adjxd][adjyd] ==0) or 
                    (mat[adjxd][adjyd*(-1)] ==0) or
                    (mat[adjxd*(-1)][adjyd] ==0) or
                    (mat[adjxd*(-1)][adjyd*(-1)] ==0)):
                        euc[x][y] = dis_i_need(x,adjxd,adjyd,y)
            else: 
                a = a +1 
                q.append[x,y]

    print("____________________")
    print(mat)
    print(euc)
#    return np.sqrt(dist)
    #print(euc)
#mat = mdup_data
print(pre_built(mat))
#print(nearestOne(mat))
    
#    return np.sqrt(dist)

# %%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(pre_built(mat))
plt.subplot(1,2,2)
plt.imshow(nearestOne(mat))
plt.show()
#%%
x = 5
y = 5
a = np.zeros(shape = (x,y))

m = a.shape[0]
n = a.shape[1]
distances = []
for x in range(m): 
    for y in range(n):
        #print(x,y)
        #distances.append((x**2 + y**2))
        a[x][y]= ((x-2)**2 + (y-2)**2)
        #a[x][y] = x,y
#        array[x][y][1] = y-2
#print(sorted(distances))
#print(array)
print(a)


# %%
def nearest(mat):

#%%


    mat = easy
    print(mat)
    MAX = 1000
    INT_MAX = 9999
    m = mat.shape[0]
    n = mat.shape[1]
    dist = np.zeros(shape = mat.shape)
    euc = np.zeros(shape = mat.shape)
    
    # two array when respective values of newx and
    # newy are added to (i,j) it gives up, down,
    # left or right adjacent of (i,j) cell
    newx = [-1, 0, 1, 0]#, -1,1,1,1]
    newy = [0, -1, 0, 1]#,-1,1,-1,1]
    #newyd= 
    

    # queue of pairs to store nodes for bfs
    q = []
     
    # traverse matrix and make pair of indices of
    # cell (i,j) having value '1' and push them
    # in queue
    for i in range(m):
        for j in range(n):
            dist[i][j] = INT_MAX
            euc[i][j] = INT_MAX
            
            if (mat[i][j] == 0):
                # distance of '0' from itself is always 0
                dist[i][j] = 0
                euc[i][j] = 0
            #else:
                q.append([i, j])
    
    # q= elements i don't have an answer for 
    poped = []
    #while (len(q)):
        
    poped = q[5]
    q.pop(0)
        
    # coordinate of currently popped node
    # position of 1
    x = poped[0]
    y = poped[1]
    print("x,y : " + str((x,y)))
    
    for i in range(len(newx)):
            
        adjx = x + newx[i]
        adjy = y + newy[i]
        
        #print("adjx,adjy : " + str((adjx,adjy)),end =" ")
        if (adjx >= 0 and adjx < m and adjy >= 0 and
                adjy < n):
                if  dist[adjx][adjy] > dist[x][y] + 1:
                    print("adjx,adjy : " + str((adjx,adjy)),end =" ")
                    print(dist[adjx][adjy],end =" ")
                    print(dist[x][y],end =" ")
                    print(dist[x][y]+1,end =" ")
                    
                    #print((adjx,adjy),end =" ")
                    #dist[adjx][adjy] > print(x,y)
                    #print(dis_i_need(x,adjx,y,adjy))
                    #euc[adjx][adjy] = dis_i_need(adjx,x,adjy,y)
                    #euc[adjx][adjy] = dis_i_need(x,adjx,y,adjy)

                    dist[adjx][adjy] = dist[x][y] + 1
                    q.append([adjx, adjy])
#%%
easy = np.array([[1,0,0,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1]])

#def nearest(mat):
#%%
    mat = easy
    #print(mat)
    MAX = 1000
    INT_MAX = 9999
    m = mat.shape[0]
    n = mat.shape[1]
    dist = np.zeros(shape = mat.shape)
    euc = np.zeros(shape = mat.shape)
    
    # two array when respective values of newx and
    # newy are added to (i,j) it gives up, down,
    # left or right adjacent of (i,j) cell
    newx = [-1, 0, 1, 0]#, -1,1,1,1]
    newy = [0, -1, 0, 1]#,-1,1,-1,1]
    #newyd= 
    

    # queue of pairs to store nodes for bfs
    q = []
     
    # traverse matrix and make pair of indices of
    # cell (i,j) having value '1' and push them
    # in queue
    for i in range(m):
        for j in range(n):
            dist[i][j] = INT_MAX
            euc[i][j] = INT_MAX
            
            if (mat[i][j] == 0):
                # distance of '0' from itself is always 0
                dist[i][j] = 0
                x_dist[i][j] = 0
                y_dist[i][j] = 0
                
                euc[i][j] = 0
            #else:
                q.append([i, j])
    
    # q= elements i don't have an answer for 
    poped = []
    
    while (len(q)):
        poped = q[0]
        q.pop(0)
        x = poped[0]
        y = poped[1]
        if dist[x][y] == 0:
                    this = x
                    that = y
        print((this,that), end ='')
        for i in range(len(newx)):
             
            adjx = x + newx[i]
            adjy = y + newy[i]
            #print((x,y),end=' ')
                          

            if (adjx >= 0 and adjx < m and adjy >= 0 and
                adjy < n):  
                #if dist[adjx][adjy] == 0:
                 #   this = adjx
                  #  that = adjx
                #print((this,that, end ='')
                #e = np.sqrt((this-adjx)**2+(that-adjy)**2)
                #print(e, end=' ')
                e = np.sqrt((this-adjx)**2+(that-adjy)**2)
                if e < euc[adjx][adjy]:
                    euc[adjx][adjy] = e
                    
                    dist[adjx][adjy] = dist[x][y] + 1
                    q.append([adjx, adjy])
                #if e > euc[adjy][adjy]
 #               if  dist[adjx][adjy] > dist[x][y] + 1:
                    ##print(dist[adjx][adjy],end =" ")
                    #print(dist[x][y],end =" ")
                    #e = np.sqrt((x-adjx)**2+(y-adjy)**2)
                    #print("adjx,adjy: " + str((adjx,adjy)),end =" ")
                                       
                    #dist[adjx][adjy] > print(x,y)
                    #print(dis_i_need(x,adjx,y,adjy))
                    #euc[adjx][adjy] = dis_i_need(adjx,x,adjy,y)
                    #euc[adjx][adjy] = dis_i_need(x,adjx,y,adjy)

                  #  dist[adjx][adjy] = dist[x][y] + 1
                   # q.append([adjx, adjy])
                    #euc[adjx][adjy] = e
                    
                    #print(adjx,adjy,x,y)
                    #np.sqrt(((x-adjx)**2)+((y-adjy)**2))
    print(dist)
    print(euc)
    print(pre_built(mat))
    #return(euc)
    #return np.sqrt(dist)
    #print(euc)
#%%
#mat = mdup_data
#print(pre_built(mat))
#print(nearest(mat))
    
plt.figure()
plt.subplot(1,2,1)
plt.imshow(pre_built(mat))
plt.subplot(1,2,2)
plt.imshow(nearest(mat))
plt.show()


# %%
easy = np.array([[0,0,0,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1],
                 [1,1,1,1,1]])

#def nearest(mat):
#%%
    mat = easy
    #print(mat)
    MAX = 1000
    INT_MAX = 999
    m = mat.shape[0]
    n = mat.shape[1]
    dist = np.zeros(shape = mat.shape)
    euc = np.zeros(shape = mat.shape)
    x_dist = np.zeros(shape = mat.shape)
    y_dist = np.zeros(shape = mat.shape)

    newx = [-1, 0, 1, 0]#, -1,1,1,1]
    newy = [0, -1, 0, 1]#,-1,1,-1,1]
    
    q = []

    for i in range(m):
        for j in range(n):
            dist[i][j] = INT_MAX
            euc[i][j] = INT_MAX
            x_dist[i][j] = INT_MAX
            y_dist[i][j] = INT_MAX

           
            if (mat[i][j] == 0):
                # distance of '0' from itself is always 0
                dist[i][j] = 0
                x_dist[i][j] = 0
                y_dist[i][j] = 0
                
                euc[i][j] = 0
            #else:
                q.append([i, j])
    
    # q= elements i don't have an answer for 
    poped = []
    #q = [[0,1],[0,2]]

    while (len(q)):
        poped = q[0]
        q.pop(0)
        x = poped[0]
        y = poped[1]
        for i in range(len(newx)):
            adjx = x + newx[i]
            adjy = y + newy[i]              
            qw = adjx
            qe = adjy
            #print((qw,qe),end=' ')
            #print(adjx,adjy)
            
            if (adjx >= 0 and adjx < m and adjy >= 0 and
                adjy < n):
                if dist[adjx][adjy]> dist[x][y] +1:
                    dist[adjx][adjy]> dist[x][y] +1
                    q.append([qw,qe])
                #if newx[i] != 0 and  x_dist[adjx][adjy] > x_dist[x][y] + 1:
                 #   x_dist[adjx][adjy] = x_dist[x][y] + 1
                    #qw = adjx
                    #q.append([qw, adjy])
                #if newy[i] != 0 and y_dist[adjx][adjy] > y_dist[x][y] + 1:
                 #   y_dist[adjx][adjy] = y_dist[x][y] + 1
                    #qe = adjy
                    #q.append([adjx, qe])

                
    print(dist)
#    print(x_dist)
 #   print(y_dist)


    for i in range((m)):
        for j in range((n)):
            euc_dist[i][j] = np.sqrt((x_dist[i][j])**2+(y_dist[i][j])**2)
    
    print(euc_dist)
    #print(euc)
    print(pre_built(mat))
    #return(euc)
    #return np.sqrt(dist)
    #print(euc)
#%%
#mat = mdup_data
#print(pre_built(mat))
#print(nearest(mat))
    
plt.figure()
plt.subplot(1,2,1)
plt.imshow(pre_built(mat))
plt.subplot(1,2,2)
plt.imshow(nearest(mat))
plt.show()
