"""
Early try implemnetation 
#################################################
comments :09/01/22

has been updated, and later version is:
try_works_ish 

can delete later
"""


#euc still not fully working
#but the rest working with input arrays

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

#%%
import numpy as np

from task import print_properties
easy = np.array([
    [0,0,1,1,0],
    [0,1,1,0,1],
    [1,1,1,1,0],
    [0,1,1,1,0],
    [0,0,1,0,0]])

MAX = 1000
INT_MAX = (2**32)
 
# distance matrix which stores the distance of
# nearest '1'

#%% 
# Function to find the nearest '1'
dist = [[0 for i in range(MAX)] for j in range(MAX)]
euc = [[0 for i in range(MAX)] for j in range(MAX)]


#%%
def nearestOne(mat, m, n):
     
    dist = np.zeros(shape = mat.shape)
    euc = np.zeros(shape = mat.shape)
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

print(nearestOne(easy,5,5))
"""
print('\n')
print(easy)
#new = easy
print(dist)
print(euc)
new = np.sqrt(dist)
print(new)
"""
#%%
"""
for i in range(5):
    for j in range(5):
        print(dist[i][j], end=" ")
        #print(euc[i][j], end=" ")
    print()
#print(np.sqrt(25))

for i in range(5):
    for j in range(5):
        print(np.sqrt(dist[i][j]), end =' ')
        #print(dist[i][j], end=" ")
        #print(new[i][j], end=" ")
    print()

for i in range(5):
    for j in range(5):
        #print(np.sqrt(dist[i][j]), end =' ')
        #print(dist[i][j], end=" ")
        print(euc[i][j], end=" ")
    print()"""
# %%



# %%
plot_comp(easy)

plot_comp(new)

# %%
