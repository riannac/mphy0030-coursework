"""
task 1 that:

working mostly, but does not give the most corrent results 
#################################################
comments :09/01/22

"""

#but the rest working with input arrays
# working in 2D!!!!!!!!!!!!!!!!!

#%%
#%%
import numpy as np
import urllib.request
from numpy.core.numeric import indices
from numpy.lib.index_tricks import index_exp
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_bf
from skimage.io import imread,imsave
from matplotlib import pyplot as plt
import extnl_code
import timeit

#%%
""" 
download label_train00.npy from download link provided in course work
“label_train00.npy” loaded as lbt_data contains a binary segmentation of Pelvic MR volume image
"""

lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)

#%%
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
    plt.figure()
    plt.subplot(1,2,1)
    if data.ndim ==3:
        data = data[slice_index, : ,:] 
    plt.imshow(data, cmap=plt.cm.gray)
    plt.subplot(1,2,2)
    plt.imshow(pre_built(data))
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

from task import print_properties
easy = np.array([
    [0,0,1,1,0],
    [0,1,1,0,1],
    [1,1,1,1,0],
    [0,1,1,1,0],
    [0,0,1,0,0]])

print_properties(easy)
print(easy.shape[1])
#%%
MAX = 1000
INT_MAX = (2**32)
 
# distance matrix which stores the distance of
# nearest '1'

#%% 
# Function to find the nearest '1'
dist = [[0 for i in range(MAX)] for j in range(MAX)]
euc = [[0 for i in range(MAX)] for j in range(MAX)]


#%%
def nearestOne3d(mat):
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
                #print("adjx,adjy" + str((adjx,adjy)), end = ' ')
                #print("x,y" + str((x,y)), end = ' ') 
                #adj -> positions of 0s
                #x,y -> coorsponding 1
                # update distance
                dist[adjx][adjy][adjz] = dist[x][y][z] + 1
                q.append([adjx, adjy, adjz])
                #euc[adjx][adjy] = np.sqrt(((x-adjx)**2)+((y-adjy)**2))
    
    return np.sqrt(dist)

def nearestOne(mat):
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
                euc[adjx][adjy] = np.sqrt(((x-adjx)**2)+((y-adjy)**2))
    
    return np.sqrt(dist)
#print(nearestOne(easy))
# #%%
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

#print_properties(easy3d)

#print(easy3d.shape[0])
#print(easy3d.shape[1])
#print(easy3d.shape[2])
#nearestOne(easy3d)
plt.figure()
#plt.subplot(1,2,1)
plt.imshow
for i in [0,1,2]:
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(easy3d[i,:,:])
    plt.subplot(1,4,2)
    plt.imshow(nearestOne3d(easy3d)[i,:,:])
    plt.subplot(1,4,3)
    plt.imshow(pre_built(easy3d)[i,:,:])
    plt.subplot(1,4,4)
    plt.imshow(nearestOne(easy3d[i,:,:]))
    plt.show()
#%%
for i in [0,1,2]:
    plt.subplot(1,2,1)
    plt.imshow(pre_built(easy3d[i,:,:]))
    plt.subplot(1,2,2)
    plt.imshow(pre_built(easy3d)[i,:,:])
    plt.show()





# %%
plot_comp(easy)
plot_comp(nearestOne(easy))

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
plot_comp(mdup_data)
plot_comp(nearestOne(mdup_data))

# %%
for n in [10,15,20,25]: 
    alg_comp(lbt_data,n)
#plot_comp(nearestOne(lbt_data[15,:,:]))
# %%
