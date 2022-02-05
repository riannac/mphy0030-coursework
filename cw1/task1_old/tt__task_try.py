"""created python file for task 1"""

"""

Original task file
now : task_try
filename: tt__task_try
#################################################
comments :09/01/22

don't think this does much special 
trying to get an implementation to work

implement distance_transform_np
input: 3D volumetric bianary image
output: 3d euclidean distance transform

"""
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

def ecld_dist_alg(p,q):
    """euclidean distance between two points
       np.sqrt((p[x]-qx)**2 + (py-qy)**2 + (pz-qz)**2)
    """
    #return np.linalg.norm(p-q)
    return np.sqrt(np.sum((p-q)**2))# axis = 1))

###########################################################################


#%%    
"""print snapshots"""
# image and it's distance transform
for n in [10,15,23]: 
    plot_comp(lbt_data, n)

# %%
#starting to try to code
"""
Euclidean transform
input = bianary image
output = distance map

each pixel contains the euclidean distance
to the closest obstacle pixel, this case boundary pixel

ecld_dist = sqrt((p[i]-q[i]**2)+(p[j]-q[j]**2)+(p[k]-q[k]**2)))

for each 1, the distance from the nearest 0
"""
# %%
###########################################################################
#                         sample arrays
##########################################################################

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
print_properties(mdup_data)

easy = np.array([
    [0,0,1,1,0],
    [0,1,1,0,1],
    [1,1,1,1,0],
    [0,1,1,1,0],
    [0,0,1,0,0]])
plot_comp(easy)
print_properties(easy)

#%%

a = np.array([[0,0,0]])
b = np.array([[1,1,1]])
#print_properties(a)
print(ecld_dist_alg(a,b))

""" 
input = input
sampling = none
return_distances = true
retrun_indices = false
distances = none
indicies = none
###
"""
def from_scipy(input,indices = None, distances =None):
    ft_inplace = isinstance(indices, np.ndarray)
    dt_inplace = isinstance(distances, np.ndarray)

    input = np.atleast_1d(np.where(input,1,0).astype(np.int8))
    
    if ft_inplace:
        ft = indices
    else: 
        ft = np.zeros((input.ndim,) + input.shape, dtype=np.int32)
   
   
    dt = ft - np.indices(input.shape, dtype=ft.dtype)
    dt = dt.astype(np.float64)
    np.multiply(dt,dt,dt)

    if dt_inplace:
        dt=np.add.reduce(dt,axis=0)
        np.sqrt(dt,distances)
    else:
        dt = np.add.reduce(dt, axis=0)
        dt = np.sqrt(dt)

    result=[]

    if not dt_inplace:
        result.append(dt)
    
    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None


# %%
print(easy)
print("_______________________________________________")
print(from_scipy(easy))
print("_______________________________________________")
print(pre_built(easy))
# %%
import numpy
def distance_transform_edt(input, sampling=None, return_distances=True,
                           return_indices=False, distances=None, indices=None):
    
    # calculate the feature transform
    input = numpy.atleast_1d(numpy.where(input, 1, 0).astype(numpy.int8))
    ft = numpy.zeros((input.ndim,) + input.shape, dtype=numpy.int32)
    print("4")
    #_nd_image.euclidean_feature_transform(input, sampling, ft)
    # if requested, calculate the distance transform
    if return_distances:
        dt = ft - numpy.indices(input.shape, dtype=ft.dtype)
        dt = dt.astype(numpy.float64)
        numpy.multiply(dt, dt, dt)
        print("5")
        dt = numpy.add.reduce(dt, axis=0)
        dt = numpy.sqrt(dt)
        print("7")

    _task_.euclidean_feature_transform(input, sampling, ft)
    # construct and return the result
    result = []
    if return_distances:
        result.append(dt)
        print("8")

    if len(result) == 2:
        print("10")
        return tuple(result)
    elif len(result) == 1:
        print("11")
        return result[0]
    else:
        print("12")
        return None 


# %%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(brute(easy), cmap=plt.cm.gray)
plt.subplot(1,2,2)
plt.imshow(pre_built(easy))
plt.show()
#%%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(distance_transform_edt(easy), cmap=plt.cm.gray)
plt.subplot(1,2,2)
plt.imshow(from_scipy(easy))
plt.show()
# %%
# Python3 program to find the minimum distance from a
# "1" in binary matrix.
MAX = 1000
INT_MAX = (2**32)
# distance matrix which stores the distance of
# nearest '1'
dist = [[0 for i in range(MAX)] for j in range(MAX)]
euc = [[0 for i in range(MAX)] for j in range(MAX)] 
# Function to find the nearest '1'
def nearestOne(mat, m, n):
    # queue of pairs to store nodes for bfs
    q = []
    yes_zero =[]
    for i in range(m):
        for j in range(n):
            dist[i][j] = INT_MAX
            if (mat[i][j] == 1):
                dist[i][j] = 0
                euc[i][j] = 0 
                # make pair and push it in queue
                q.append([i, j])
            else: 
                yes_zero.append([i,j])
    
    newx = [-1, 0, 1, 0]
    newy = [0, -1, 0, 1]
    for i in range(len(yes_zero)):
        x = yes_zero[i][0]
        y = yes_zero[i][1]
        #print(i, x, y)
        for i in range(4):
                adjx = x + newx[i]
                adjy = y + newy[i]
              

                if (adjx >= 0 and adjx < m and adjy >= 0 and
                    adjy < n and dist[adjx][adjy] > dist[x][y] + 1):
                    
                    # update distance
                    dist[adjx][adjy] = dist[x][y] + 1
                    print(adjx,adjy)
                    q.append([adjx, adjy])
                    euc[x][y] = np.sqrt(((x-y)**2)+((adjx-adjy)**2))
            
    """
    # now do bfs traversal
    # pop element from queue one by one until it gets empty
    # pair element to hold the currently popped element
    poped = []
    for one in not_zero:
        x = not_zero[0]
        y = not_zero[1]

        print(x,y)"""
        
    #print(yes_zero)
    #print(yes_zero[1][1])
nearestOne(easy,5,5)
print(easy)

#%%
    while (len(q)):
        poped = q[0]
        q.pop(0)
         
        # coordinate of currently popped node
        x = poped[0]
        y = poped[1]
         
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
                 
                # update distance
                dist[adjx][adjy] = dist[x][y] + 1
                q.append([adjx, adjy])
                euc[x][y] = np.sqrt(((x-y)**2)+((adjx-adjy)**2))
                
                
# %%
nearestOne(easy, 5, 5)
print(easy)
for i in range(5):
    for j in range(5):
        #print(dist[i][j], end=" ")
        print(euc[i][j], end=" ")
    print()
#print(pre_built(easy)    )
"""
for i in range(5):
    for j in range(5):
        print(dist[i][j], end=" ")
        #print(euc[i][j], end=" ")
    print()"""
 
# %%

# %%
#%%
import numpy as np
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

def nearestOne(mat, m, n):
     
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
            if (mat[i][j] == 1):
                 
                # distance of '1' from itself is always 0
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
        # position of 1
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
                print("adjx,adjy" + str((adjx,adjy)), end = ' ')
                print("x,y" + str((x,y)), end = ' ') 
                #adj -> positions of 0s
                #x,y -> coorsponding 1
                # update distance
                dist[adjx][adjy] = dist[x][y] + 1
                q.append([adjx, adjy])
                euc[adjx][adjy] = np.sqrt(((x-adjx)**2)+((y-adjy)**2))
                

nearestOne(easy,5,5)
print('\n')
#print(easy)
#new = easy
for i in range(5):
    for j in range(5):
        print(dist[i][j], end=" ")
        #print(euc[i][j], end=" ")
    print()
#print(np.sqrt(25))
print_properties(dist)
print_properties(easy)
"""
for i in range(5):
    for j in range(5):
        new = (np.sqrt(dist[i][j]))
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
