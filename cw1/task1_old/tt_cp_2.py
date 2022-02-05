"""
Early task copy
#################################################
comments :09/01/22
containing functions: 

can delete later
"""
#%%
import numpy as np
import urllib.request
from numpy.lib.index_tricks import index_exp
from scipy import ndimage
from skimage.io import imread,imsave
from matplotlib import pyplot as plt
import extnl_code

#%%
""" 
download label_train00.npy from download link provided in course work
“label_train00.npy” loaded as lbt_data contains a binary segmentation of Pelvic MR volume image
"""

lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)

#%%
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

#%%    
#d_trans_edt = pre_built(lbt_data)
#print_properties(lbt_data)
#print_properties(d_trans_edt)

# image and it's distance transform
for n in [10,15,23]: 
    plot_comp(lbt_data, n)

# %%
#Making up a smaller bianry array to test distance transform on
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
"""
with np.nditer(mdup_data, op_flags=['readwrite']) as it:
    for x in it:
        x[...] = 2 * x

it = np.nditer(mdup_data, flags=['multi_index'])
print(mdup_data)
for x in it:
    print("%d <%s>" % (x, it.multi_index), end='')
    
    for i in range(15):
    for j in range(15):
        print(mdup_data[i,j], end=' ')
        print([i,j], end=' ')

"""
easy = np.array([
    [0,0,1,1,0],
    [0,1,1,0,1],
    [1,1,1,1,0],
    [0,1,1,1,0],
    [0,0,1,0,0]])
plot_comp(easy)
print_properties(easy)

#%%
#store each i, j, k, in separate array
#for each in range(easy[:,1]):
print (easy)
it = np.nditer(easy, flags=['multi_index'])
indexes = np.zeros(easy.shape)
for x in it:
    c = c+ 1
    indexes[c] = (it.multi_index)
    print("%d <%s>" % (x, it.multi_index), end='')
#print(indexes)
#r = mdup_data*2
#print(r)


#%%
# difference euc distances array 
b = easy.reshape(np.prod(easy.shape[:-1]),1,easy.shape[-1])
diff = easy - b; dist_arr = np.sqrt(np.einsum('ijk,ijk->ij',diff,diff))
print(pre_built(easy))
print(dist_arr)


def ecld_dist_alg(p,q):
    """euclidean distance algorithm"""
    
    np.sqrt((p[x]-qx)**2 + (py-qy)**2 + (pz-qz)**2)