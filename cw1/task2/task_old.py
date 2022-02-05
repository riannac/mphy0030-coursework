"""
Suface_normals_np 
takes triangulated surface as input, represented by
a list of verices and a list of triangles. 
And returns two type of normal vectors
1) at vertices and 
2) at triangle centres 

"""
#%%
import numpy as np
from numpy.core.fromnumeric import shape 
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import urllib.request

#%%
""" 
download label_train00.npy from download link provided in course work
“label_train00.npy” loaded as lbt_data contains a binary segmentation of Pelvic MR volume image
"""

lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)
vox_dims = (2,.5,.5)

#%%
mdup_data = np.array([
                 [[0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0],
                 [0, 0, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 0, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 0, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 0, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 0, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,0],
                 [0, 0, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,0,0],
                 [0, 0, 1, 1, 1, 1, 1, 1, 1,1,1,1,0,0,0],
                 [0, 0, 1, 0, 0, 1, 1, 1, 1,0,0,0,0,0,0],
                 [0, 0, 1, 0, 0, 0, 1, 1, 1,1,0,0,0,0,0],
                 [0, 0, 1, 0, 0, 0, 0, 1, 1,1,1,0,0,0,0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 1,1,1,1,0,0,0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0,1,1,1,1,0,0],
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

def print_properties(x):
    """prints the properties of an array"""
    print("\n type : " + str(type(x))
    + "\n size : " + str(x.size)
    + "\n shape : " + str(x.shape)
    + "\n dtype : " + str(x.dtype)
    + "\n ndim : " + str(x.ndim)
    + "\n itemsize : " + str(x.itemsize)
    + "\n nbytes : " + str(x.nbytes))

print_properties(lbt_data)

#%%
"""
At vertices: averageing the normal of all the triangles
one vertex is a part of
"""
#%%
def normalize_v3(arr):    
    """Normalize a numpy array of 3 component vectors shape=(n,3)"""    
    lens = np.sqrt(arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2)     
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                     
    return arr

def normals_mine(vertices, faces):
    """
    Create a zeroed array with the same type and shape as
    our vertices i.e. per vertex normal 
    """
    norm = np.zeros(vertices.shape) 
    """
    Create an indexed view into the vertex array using the
    array of three indices for trianges 
    """
    tris = vertices[faces]
    """
    calculate the normal for all the trianges, 
    by taking the cross product of the vectors v1-v0, v2-v0 in 
    each triangle """
    n = np.cross(tris[::,1]-tris[::,0], tris[::,2]-tris[::,0])
    #print(n)
    #print(n.shape)
    # n is now an array of normals per triangle. 
    # The length of each normal is dependent the vertices, 

    # we need to normalize these, 
    # so that our next step weights each normal equally.

    normalize_v3(n) 
    #print(n)
    #print(n[1])
    # now we have a normalized array of normals, 
    # one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), 
    # we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then 
    # contribute to every vertex, so we need to normalize again
    # afterwards.
    # The cool part, we can actually add the normals through 
    # an indexed view of our (zeroed) per vertex normal 
    # array 
    norm[ faces[:,0] ] += n 
    norm[ faces[:,1] ] += n 
    norm[ faces[:,2] ] += n
    normalize_v3(norm)
    
    #va = vertices[faces] 
    #no = norm[faces]
    return(norm)
    #return va, no
    

verts, faces, normals, values = marching_cubes(mdup_data)

nm = abs(normals_mine(verts, faces))
abs(nm)
print(nm.shape)
#print(nm[3])
#print(verts.shape)
#print(faces.shape)

print(normals.shape)
#print(normals[3])

fig = plt.figure()
ax = fig.gca(projection='3d')
px = verts
trix = faces
#plt.subplot(1,2,1)
ax.plot_trisurf(px[:,0], px[:,1], px[:,2], triangles=trix, color='b')
ax.quiver(verts[:,0],verts[:,1],verts[:,2], nm[:,0],nm[:,1],nm[:,2])
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
px = verts
trix = faces
ax.plot_trisurf(px[:,0], px[:,1], px[:,2], triangles=trix, color='b')
ax.quiver(verts[:,0],verts[:,1],verts[:,2], normals[:,0],normals[:,1],normals[:,2])
plt.show()

#%%
verts, faces, normals, values = marching_cubes(mdup_data,spacing=vox_dims[0:3],step_size=2)
print(normals)
#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, projection='3d')
#mesh = Poly3DCollection(verts[faces])
#mesh.set_edgecolor('k')
#ax.add_collection3d(mesh)
#plt.tight_layout()
#plt.show()

# %%
# call guassian filter on image
# normalised 0-255
# int unsinged 8
# and smooth it

