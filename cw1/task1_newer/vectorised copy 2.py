#%%
import numpy as np
import time
import urllib.request
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_bf
from matplotlib import pyplot as plt
#%%
#setting arrays
easy = np.array([
    [1,0,0,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,1],
    [0,1,1,1,1]])

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


mdup_data3d = np.array([
                [[0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0]],
                 
                 [[0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 1, 1, 1, 1,0,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 1, 1, 1,1,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 1, 1,1,1,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,1,1,1,1,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0]],

                 [[0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 1, 1, 1, 0, 0, 0,1,1,1,1,1,0],
                 [0, 1, 1, 1, 0, 0, 0, 0, 0,0,0,1,1,1,0],
                 [0, 1, 1, 1, 1, 1, 0, 0, 0,1,1,1,1,1,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 1, 1, 1, 1,0,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 1, 1, 1,1,0,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 1, 1,1,1,0,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 1,1,1,1,0,0,0],
                 [0, 1, 1, 0, 0, 0, 0, 0, 0,1,1,1,1,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0]],

                 [[0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
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
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0]],

                
                 ])


#%%
#function definitions

def print_properties(x):
    print("\n type : " + str(type(x))
    + "\n size : " + str(x.size)
    + "\n shape : " + str(x.shape)
    + "\n dtype : " + str(x.dtype)
    + "\n ndim : " + str(x.ndim)
    + "\n itemsize : " + str(x.itemsize)
    + "\n nbytes : " + str(x.nbytes))
    
# visualisatino while developing comparing image and
# it's transform and specified slices
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

def brute(vol_bi_img):
    return ndimage.distance_transform_bf(vol_bi_img)

###########################################################################

#%%
#Download data
""" 
download label_train00.npy from download link provided in course work
“label_train00.npy” loaded as lbt_data contains a binary segmentation of Pelvic MR volume image
"""

lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)


#%%
"""
Euclidean transform
input = bianary image
output = distance map

each pixel contains the euclidean distance
to the closest obstacle pixel, this case boundary pixel

ecld_dist = sqrt((p[i]-q[i]**2)+(p[j]-q[j]**2)+(p[k]-q[k]**2)))

for each 1, the distance from the nearest 0
"""

def distance_transform_np(v_b_i):
    """docstring to describe the algorithm used"""
    #v_b_i = volumetric binary image
    zeros_coord = np.where(v_b_i == 0) #find the coordinates of all zeros 
    zeros_ = np.asarray(zeros_coord, dtype=np.int16).T # store them
    ones_coord = np.where(v_b_i == 1) # find coordinates of all ones
    ones_ = np.asarray(ones_coord, dtype=np.int16).T #store them

    # calculate distance 
    # sum of coordinates squared
    a = -2 * np.dot(ones_, zeros_.T) 
    b = np.sum(np.square(zeros_), axis=1, dtype=np.int16) 
    c = np.sum(np.square(ones_), axis=1, dtype=np.int16)[:,np.newaxis]
    euc_dist = a + b + c
    # sqrt of sums
    # min euclidean dist of each one pixel to zero pixel
    euc_dist = np.sqrt(euc_dist.min(axis=1)) 
    x = v_b_i.shape[0]
    y = v_b_i.shape[1]
    z = v_b_i.shape[2]
    euc_d_transform = np.zeros((x,y,z), dtype=np.int16)
    euc_d_transform[ones_[:,0], ones_[:,1], ones_[:,2]] = euc_dist 
    #print(euc_d_transform.dtype)
    #ne = euc_d_transform.astype(np.int8)
    #print(ne.dtype)
    
    #plt.figure()
    return (euc_d_transform)


def get_dst_transform_img_(og): #og is a numpy array of original image
    print(og.dtype)
    ones_loc = np.where(og == 0)
    ones = np.asarray(ones_loc).T # coords of all ones in og
    zeros_loc = np.where(og == 1)
    zeros = np.asarray(zeros_loc).T # coords of all zeros in og

    a = -2 * np.dot(zeros, ones.T) 
    b = np.sum(np.square(ones), axis=1) 
    c = np.sum(np.square(zeros), axis=1)[:,np.newaxis]
    dists = a + b + c
    dists = np.sqrt(dists.min(axis=1)) # min dist of each zero pixel to one pixel
    x = og.shape[0]
    y = og.shape[1]
    z = og.shape[2]
    dist_transform = np.zeros((x,y,z))
    dist_transform[zeros[:,0], zeros[:,1], zeros[:,2]] = dists 
    #print(dist_transform.dtype)
    #ne = dist_transform.astype(np.int8)
    #print(ne.dtype)
    
    #plt.figure()
    return (dist_transform)

def get_dst_transform_img(og): #og is a numpy array of original image
   ones_loc = np.where(og == 0)
   ones = np.asarray(ones_loc).T # coords of all ones in og
   zeros_loc = np.where(og == 1)
   zeros = np.asarray(zeros_loc).T # coords of all zeros in og

   a = -2 * np.dot(zeros, ones.T) 
   b = np.sum(np.square(ones), axis=1) 
   c = np.sum(np.square(zeros), axis=1)[:,np.newaxis]
   dists = a + b + c
   dists = np.sqrt(dists.min(axis=1)) # min dist of each zero pixel to one pixel
   x = og.shape[0]
   y = og.shape[1]
   z = og.shape[2]
   dist_transform = np.zeros((x,y,z), dtype =np.float16)
   dist_transform[zeros[:,0], zeros[:,1], zeros[:,2]] = dists 

   
   return dist_transform

#%%

#%%
f = 7 
g = 14
a = lbt_data[f:g]

j=19
k=25
b = lbt_data[j:k]

a11 = pre_built(lbt_data)

my_alg_time = time.time()
a1 = pre_built(a)
b1 = pre_built(b)
print("pre_build takes %s seconds " % (time.time() - my_alg_time))

#print(a1[15])
pre_alg_time = time.time()
a2 = get_dst_transform_img(a)
#a2 = pre_built(a)
print("new takes %s seconds " % (time.time() - pre_alg_time))

#pre_alg_time = time.time()
#a3 = distance_transform_np(a)
a3 = get_dst_transform_img_(a)
print(a3.dtype)
#a2 = pre_built(a)
#print("best takes %s seconds " % (time.time() - pre_alg_time))
#print(a1)
#print(a2)

#print(np.array_equal (a11[f], a2[1], equal_nan=False))
#print(np.allclose(a1,a2, rtol=.2, atol=.2))

plt.figure()
plt.subplot(1,3,1)
plt.imshow(a1[1])
plt.subplot(1,3,2)
plt.imshow(a2[1])
plt.subplot(1,3,3)
plt.imshow(a3[1])
plt.show()

"""

for i in [0,1,2,5,7]:
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(a11[f+i])
    plt.subplot(1,3,2)
    plt.imshow(a2[i])
    plt.subplot(1,3,3)
    plt.imshow(a[i])
    plt.show()
"""
#%%
from PIL import Image
#myarray= a1[15]
from matplotlib import cm
for i in [0,1,3,4]:
    myarray=a11[g+i]
    im = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))
    title = 'pre_built_slice_' + str(i) +'.png'
    im.save(title)
    myarray2=a2[i]
    im = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))
    title = 'mine_vector_slice_' + str(i) +'.png'
    im.save(title)
#
#
#PIL_image = Image.fromarray(np.uint8(numpy_image)).convert('RGB')

#PIL_image = Image.fromarray(numpy_image.astype('uint8'), 'RGB')

# %%
