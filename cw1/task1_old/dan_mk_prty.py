
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
from numpy.lib.function_base import append
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_bf
from matplotlib import pyplot as plt

#importing data

""" 
download label_train00.npy from download link provided in course work
“label_train00.npy” loaded as lbt_data contains a binary segmentation of Pelvic MR volume image
"""

lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)


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


#build my try algorithms
#def dans(mat):
#%%
    mat = easy
    m = mat.shape[0]
    n = mat.shape[1]
    
    # initiate lists to store index information
    ones=[]
    zeros=[]
    
    # initiate output array
    euc_array = np.zeros(shape=(mat.shape))

    # iterate through every point in the matrix
    # if point = 0, add its index to zeros
    # if point = 1, add its index to ones
    ones = np.nonzero(mat)

    print(ones)
#%%

    for i in range(m):
        for j in range(n): 
            ones.append(np.array([i,j]))

    # for every 1, caluclate the distance to each 0
    # and store it in a list
    # set the min value of that list equal to the 
    # index on the coresponding 1 in the output array 
    for one in ones:
        euc_dist=[]
        a = 0
        for zero in zeros:
            dist = np.sqrt((one[0]-zero[0])**2+(one[1]-zero[1])**2)
            euc_dist.append(dist)
            if euc_dist[a-1] < euc_dist[a]:

            a += 1
        euc_array[one[0]][one[1]] = min(euc_dist)
        
    return euc_array

#3d version
def dans3d(mat, voxel_dims = 1):
    m = mat.shape[0]
    n = mat.shape[1]
    l = mat.shape[2]
    # initiate lists to store index information
    ones=[]
    zeros=[]
    # initiate output array
    euc_array = np.zeros(shape=(mat.shape))

    # iterate through every point in the matrix
    # if point = 0, add its index to zeros
    # if point = 1, add its index to ones
    for i in range(m):
        for j in range(n): 
            for k in range(l):
                if mat[i][j][k]==0:
                    zeros.append(np.array([i,j,k]))
                else: 
                    ones.append(np.array([i,j,k]))
    # for every 1, caluclate the distance to each 0
    # and store it in a list
    # set the min value of that list equal to the 
    # index on the coresponding 1 in the output array 
    for one in ones:
        euc_dist=[]
        for zero in zeros:
            dist = np.sqrt((one[0]-zero[0])**2+(one[1]-zero[1])**2+(one[2]-zero[2])**2)
            euc_dist.append(dist)
        euc_array[one[0]][one[1]][one[2]] = min(euc_dist)
        
    return euc_array

a =lbt_data #[1:6][1:5][1:5]
#print_properties(a)

#a = easy3d
my_alg_time = time.time()
a1 = pre_built(a)
print("pre_build takes %s seconds " % (time.time() - my_alg_time))

#print(a1[15])
pre_alg_time = time.time()
a2 = dans3d(a)
#a2 = pre_built(a)
print("mine takes %s seconds " % (time.time() - pre_alg_time))


# values
#print(np.array_equal (a1, a2, equal_nan=False))
#print(np.allclose(a1,a2, rtol=.2, atol=.2))

#ploting
from PIL import Image
#myarray= a1[15]
from matplotlib import cm
for i in [15,20,23]:
    myarray=a1[i]
    im = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))
    title = 'pre_built_slice_' + str(i) +'.png'
    im.save(title)
#
#PIL_image = Image.fromarray(np.uint8(numpy_image)).convert('RGB')

#PIL_image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')

"""
for i in [15,20,25]:
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(a1[i])
    #plt.subplot(1,2,2)
    #plt.imshow(a2[i])
    plt.show()
"""


