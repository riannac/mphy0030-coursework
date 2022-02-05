"""
Suface_normals_np 
takes triangulated surface as input, represented by
a list of verices and a list of triangles. 
And returns two type of normal vectors
1) at vertices and 
2) at triangle centres 

"""
#%%
from cmath import cos
import numpy as np
from numpy.core.fromnumeric import shape 
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import urllib.request

#%%
""" 
download label_train00.npy from download link provided in course work
“label_train00.npy” loaded as lbt_data contains a binary segmentation of 
Pelvic MR volume image
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

#%%
"""Implement a function surface_normals_np"""
def normalised(array):    
    """Normalize an x,y,z array"""    
    avr = (array[:,0]**2 + array[:,1]**2 + array[:,2]**2)**(.5)    
    array[:,0] /= avr
    array[:,1] /= avr
    array[:,2] /= avr                     
    return array

def surface_normals_np(vertices, triangles):
    """Takes a triangulated surface as input 
    and returns two types of normal vectors,
    at vertices (vert_norm) and
    at triangle centres (cent_norm)
    """
    vert_norm = np.zeros(shape = vertices.shape) 
    tris = vertices[triangles]
    v = tris[::,2] - tris[::,0]
    u = tris[::,1] - tris[::,0]
    cent_norm = np.cross(v, u)
    normalised(cent_norm) 
    vert_norm[ triangles[:,0] ] += cent_norm 
    vert_norm[ triangles[:,1] ] += cent_norm
    vert_norm[ triangles[:,2] ] += cent_norm
    normalised(vert_norm)
    return(vert_norm, cent_norm)

def cos_compare(my_norms, mm_normals):
    """Cosine compare for two sets of normals"""
    cos_sim = np.zeros(shape = my_norms.shape[0], dtype=np.float16)
    for each in range(my_norms.shape[0]):
        a = my_norms[each]
        b = mm_normals[each]
        c = np.dot(a,b)/((np.linalg.norm(a))*(np.linalg.norm(b)))
        cos_sim[each] = c
    #print("Cosine Similarity:")
    #print(" Mean = " + str(np.mean(cos_sim)),end = '')
    #print(" Median = " +str(np.median(cos_sim)),end = '')
    #print(" Standard deviation = " + str(np.std(cos_sim)))
    print("( Mean, Median, Std ) ", end = ' ') 
    print(((np.mean(cos_sim)),(np.median(cos_sim)),(np.std(cos_sim))))


def face_from_vert(triangles, normals):
    """average of triangle normals = (x+x+x/3, y+y+y/3, z+z+z/3 etc)"""
    fa_norm = np.zeros(shape = triangles.shape)
    for tri in range(triangles.shape[0]):
        nt = normals[triangles[tri]]
        fa_norm[tri] = (nt[0]+nt[1]+nt[2])/3
        
    return(fa_norm)

#%%
"""use mearse marching cubes to compute vertex normals"""
data = lbt_data
vertex_coord, tri_angles, mm_normals, _ = marching_cubes(data,spacing=vox_dims[0:3],step_size=2)
vert_norms, cent_norms =(surface_normals_np(vertex_coord, tri_angles))

#vert_norms = normals
"""Determine a reasonable metric for comparing
average cosine similarity"""
print("\n")
cos_compare(vert_norms, mm_normals)    
print("The similarity of the results is quite high, however, due to the approximations involved " +
"in the Marching Cubes algorithm, the normals generated by surface_normals_np" + 
"are not exactly the same as those generated by Marching Cubes.")
#%%
"""design  a method to compare vertex normals and triangle centre normals"""

print("\nThere is not standard way to go from vertex to face normal, " + 
"as there is no way of doing this with out losing information." + 
" I have attempted to compare by finding the average of the vertex normals, " + 
"for each triangle, and comparing to my face normals.")

mm_face = face_from_vert(tri_angles, mm_normals)
cos_compare(cent_norms,mm_face)
#my_v2face = face_from_vert(tri_angles, vert_norms)
#cos_compare(cent_norms,my_v2face)

# %%
"""Use a gaussian filter"""
data = lbt_data
sigmas = [0.5,1,2.,2.5,4]
#sigmas = [10,20,40]
slice = 20
a = 1
rr_mmv =[]
rr_vn =[]
rr_cn = []
#rr_mmv.append(gmm_normals)
#rr_vn.append(gvert_norms)
#rr_cn.append(gcent_norms)
# call guassian filter on image
# normalised 0-255
# int unsinged 8
# and smooth it
plt.figure()
plt.subplot(1,(len(sigmas))+1,a)
plt.title("og")
plt.imshow(data[slice])
print("\nThe higher the sigma, the more smoothing there is, therefore, the image gets smaller as the outer layer is erroded.")
for s in sigmas:
    a += 1
    plt.subplot(1,(len(sigmas))+1,a)
    gdata = gaussian_filter(data,sigma=(s*.05,s*.2,s*.2),output = np.uint8)
    #rr_s.append(gdata)
    plt.title(str(s))
    plt.imshow(gdata[slice])
    gvertex_coord, gtri_angles, gmm_normals, _ = marching_cubes(gdata,spacing=vox_dims[0:3],step_size=2)
    gvert_norms, gcent_norms =(surface_normals_np(gvertex_coord, gtri_angles))
    print((s*.05,s*.2,s*.2))
    cos_compare(gmm_normals,mm_normals)
    cos_compare(gvert_norms,vert_norms)
    cos_compare(gcent_norms,cent_norms)
#%%
"""
fig = plt.figure()
#plt.subplot(1,2,1)
ax = fig.gca(projection='3d')
px = verts
trix = faces

ax.plot_trisurf(px[:,0], px[:,1], px[:,2], triangles=trix, color='b')
ax.quiver(verts[:,0],verts[:,1],verts[:,2], nm[:,0],nm[:,1],nm[:,2])
#plt.show()

fig = plt.figure()
#plt.subplot(1,2,2)
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
"""
