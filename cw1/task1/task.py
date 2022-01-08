"""created python file for task 1"""

"""
Task.py should perform: 
download label train from;
https:weisslab.cd.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy

implement distance_transform_np
input: 3D volumetric bianary image
output: 3d euclidean distance transform

"""
#%%
import numpy as np
import urllib.request
from numpy.lib.index_tricks import index_exp
from scipy import ndimage
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
    return np.sqrt(np.sum((p-q)**2, axis = 1))

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
#%%

a = np.array([[0,0,0]])
b = np.array([[1,1,1]])
#print_properties(a)
print(ecld_dist_alg(a,b))

# %%
#def idk(input):
 #   ft = np.zeroes((input.ndim,), input.shape, dt)