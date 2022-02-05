"""
Task copy
################################################
comments :09/01/22

trying to understand how the in built scipy
function works, by backbuilding using all their
in built/references functions
"""
#%%
import numpy as np
import urllib.request
from numpy.core.numeric import indices
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
def generate_binary_structure(rank, connectivity):
    if connectivity < 1:
            connectivity = 1
    if rank < 1:
        return numpy.array(True, dtype=bool)
    output = numpy.fabs(numpy.indices([3] * rank) - 1)
    output = numpy.add.reduce(output, 0)
    return output <= connectivity
def normalize_sequence(input, rank):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    normalized = [input] * rank
    return normalized
def _binary_erosion(input, structure, iterations, mask, output,
                    border_value, origin, invert, brute_force):
    try:
        iterations = operator.index(iterations)
    except TypeError as e:
        raise TypeError('iterations parameter should be an integer') from e

    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    else:
        structure = numpy.asarray(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have same dimensionality')
    if not structure.flags.contiguous:
        structure = structure.copy()
    if numpy.prod(structure.shape, axis=0) < 1:
        raise RuntimeError('structure must not be empty')
    if mask is not None:
        mask = numpy.asarray(mask)
        if mask.shape != input.shape:
            raise RuntimeError('mask and input must have equal sizes')
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    cit = _center_is_true(structure, origin)
    if isinstance(output, numpy.ndarray):
        if numpy.iscomplexobj(output):
            raise TypeError('Complex output type not supported')
    else:
        output = bool
    output = _ni_support._get_output(output, input)
    temp_needed = numpy.may_share_memory(input, output)
    if temp_needed:
        # input and output arrays cannot share memory
        temp = output
        output = _ni_support._get_output(output.dtype, input)
    if iterations == 1:
        _nd_image.binary_erosion(input, structure, mask, output,
                                 border_value, origin, invert, cit, 0)
        return output
    elif cit and not brute_force:
        changed, coordinate_list = _nd_image.binary_erosion(
            input, structure, mask, output,
            border_value, origin, invert, cit, 1)
        structure = structure[tuple([slice(None, None, -1)] *
                                    structure.ndim)]
        for ii in range(len(origin)):
            origin[ii] = -origin[ii]
            if not structure.shape[ii] & 1:
                origin[ii] -= 1
        if mask is not None:
            mask = numpy.asarray(mask, dtype=numpy.int8)
        if not structure.flags.contiguous:
            structure = structure.copy()
        _nd_image.binary_erosion2(output, structure, mask, iterations - 1,
                                  origin, invert, coordinate_list)
    else:
        tmp_in = numpy.empty_like(input, dtype=bool)
        tmp_out = output
        if iterations >= 1 and not iterations & 1:
            tmp_in, tmp_out = tmp_out, tmp_in
        changed = _nd_image.binary_erosion(
            input, structure, mask, tmp_out,
            border_value, origin, invert, cit, 0)
        ii = 1
        while ii < iterations or (iterations < 1 and changed):
            tmp_in, tmp_out = tmp_out, tmp_in
            changed = _nd_image.binary_erosion(
                tmp_in, structure, mask, tmp_out,
                border_value, origin, invert, cit, 0)
            ii += 1
    if temp_needed:
        temp[...] = output
        output = temp
    return output
def binary_dilation(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0, origin=0,
                    brute_force=False):
    input = numpy.asarray(input)
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    origin = normalize_sequence(origin, input.ndim)
    structure = numpy.asarray(structure)
    structure = structure[tuple([slice(None, None, -1)] *
                                structure.ndim)]
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1

    return _binary_erosion(input, structure, iterations, mask,
                           output, border_value, origin, 1, brute_force)

def distance_transform_bf(input, metric="euclidean", sampling=None,
                          return_distances=True, return_indices=False,
                          distances=None, indices=None):
    ft_inplace = isinstance(indices, numpy.ndarray)
    dt_inplace = isinstance(distances, numpy.ndarray)
    
    tmp1 = numpy.asarray(input) != 0
    struct = generate_binary_structure(tmp1.ndim, tmp1.ndim)
    tmp2 = binary_dilation(tmp1, struct)
    tmp2 = numpy.logical_xor(tmp1, tmp2)
    tmp1 = tmp1.astype(numpy.int8) - tmp2.astype(numpy.int8)
    metric = metric.lower()
    if metric == 'euclidean':
        metric = 1
    ft = None
    if return_distances:
        if distances is None:
            if metric == 1:
                dt = numpy.zeros(tmp1.shape, dtype=numpy.float64)     
        ft = numpy.ravel(ft)
        for ii in range(tmp2.shape[0]):
            rtmp = numpy.ravel(tmp2[ii, ...])[ft]
            rtmp.shape = tmp1.shape
            tmp2[ii, ...] = rtmp
        ft = tmp2

    # construct and return the result
    result = []
    if return_distances and not dt_inplace:
        result.append(dt)
    if return_indices and not ft_inplace:
        result.append(ft)

    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None

# %%
plot_comp(distance_transform_bf(easy))
# %%
