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
from scipy import ndimage
from skimage.io import imread,imsave
from matplotlib import pyplot as plt
import extnl_code

# download label_train00.npy from download link provided in course work
# “label_train00.npy” loaded as lbt_data contains a binary segmentation of Pelvic MR volume image
lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)

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

"""this is the in built distance transform to compare with"""
d_trans_edt = ndimage.distance_transform_edt(lbt_data)

#print_properties(lbt_data)
#print_properties(d_trans_edt)

#%%
"""visualisatino while developing comparing image and it's transform and specified slices"""
for n in [10,15,23]: 
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(lbt_data[n,:,:], cmap=plt.cm.gray)
    plt.subplot(1,2,2)
    plt.imshow(d_trans_edt[n,:,:])
    plt.show()
# %%
