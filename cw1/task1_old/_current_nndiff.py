

#############################################################
#           2d WORKING ! as of 09.01.22
#           3d WORKING ! but very very slow 
#                   and pretty
"""
                       working on it 
"""
##########################################################

# 09/01/22
# trying to just compare slices


"""trying from the start with new method""" 

"""
Task.py should perform: 
download label train from;
https:weisslab.cd.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy

implement distance_transform_np
input: 3D volumetric bianary image
output: 3d euclidean distance transform

"""
#%%
#import statements 
import numpy as np
import time
import urllib.request
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_bf
from matplotlib import pyplot as plt
#%%
#importing data

""" 
download label_train00.npy from download link provided in course work
“label_train00.npy” loaded as lbt_data contains a binary segmentation of Pelvic MR volume image
"""

lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)

#%%
#defining functions
################################################################################
#                            Function Definitions                              #
################################################################################

def print_properties(x):
    """prints the properties of an array"""
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
    """this is the scipy distance transform to compare with"""
    return ndimage.distance_transform_edt(vol_bi_img)
def brute(vol_bi_img):
    """this is the scipy brute force distance transform"""
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

def pre_built(vol_bi_img):
    """this is the in built distance transform to compare with"""
    return ndimage.distance_transform_edt(vol_bi_img)


#%%
#setting testing arrays
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
#building my function definition 
def Euc_transform_2d(input):
    N = input.shape[0]
    M = input.shape[1]
    euc = np.zeros(shape = input.shape)
    
    # Initialize the transform array
    for i in range(N):
        for j in range(M):
            euc[i][j] = 999999999999
    
    # For each cell 
    for i in range(N):
        for j in range(M):
            
            # go through the whole arrary
            # and find the min distance from
            # a cell containing 0 
            for n in range(N):
                for m in range(M):
                    
                    # for cells that contain 0, 
                    # find the minimum distance. 
                    # and save this in cooresponting 
                    if (input[n][m] == 0):
                        euc[i][j] = min(euc[i][j], 
                                    np.sqrt((((i-n)**2))+((j-m)**2)))
   
    return euc

# 09/01/22
# Removed a function
#still in the old one if needed

def Euc_transform_3d(input):
    N = input.shape[0]
    M = input.shape[1]
    L = input.shape[2]
    euc = np.zeros(shape = input.shape)
    
    # Initialize the transform array
    for i in range(N):
        for j in range(M):
            for k in range(L):    
                euc[i][j][k] = 999999999999
    
     
    # For each cell 
    for i in range(N):
        for j in range(M):
            for k in range(L):
            # go through the whole arrary
            # and find the min distance from
            # a cell containing 0 
                for n in range(N):
                    for m in range(M):
                        for l in range(L):
                            # for cells that contain 0, 
                            # find the minimum distance. 
                            # and save this in cooresponting  
                            if (input[n][m][l] == 0):
                            
                                euc[i][j][k] = min(euc[i][j][k], 
                                            np.sqrt((((i-n)**2))+((j-m)**2) +((k-l)**2)))
    return euc


#%%
# comparing mine and pre_built
# timing
my_alg_time = time.time()
a1 = Euc_transform_3d(easy3d)
print("--- %s seconds ---" % (time.time() - my_alg_time))

pre_alg_time = time.time()
a2 = pre_built(easy3d)
print("--- %s seconds ---" % (time.time() - pre_alg_time))

# values
print(np.array_equal (a1, a2, equal_nan=False))
print(np.allclose(a1,a2, rtol=.2, atol=.2))

#ploting
plt.subplot(1,2,1)
plt.imshow(a1[2])
plt.subplot(1,2,2)
plt.imshow(a2[2])
plt.show()

#%%
# comparing mine and pre_built
# for smaller sections of the data
# timing
a = lbt_data[:][:][16]
#a= mdup_data
my_alg_time = time.time()
a1 = Euc_transform_2d(a)
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

# %%
print(Euc_transform)
# %%
