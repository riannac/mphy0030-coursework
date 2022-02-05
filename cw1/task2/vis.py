#%%
import numpy as np 
import matplotlib.pyplot as plt 
#from task import surface_normals_np
from skimage.measure import marching_cubes
import urllib.request

lbt_url = 'https://weisslab.cs.ucl.ac.uk/WEISSTeaching/datasets/-/raw/promise12/label_train00.npy'
urllib.request.urlretrieve(lbt_url, 'lbt_file.npy')
lbt_data = np.load('lbt_file.npy',allow_pickle=False)
vox_dims = (2,.5,.5)

mdup_data = np.array([
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
    

data = lbt_data
vertex_coord, tri_angles, mm_normals, _ = marching_cubes(data,spacing=vox_dims[0:3],step_size=2)
vert_norms, cent_norms =(surface_normals_np(vertex_coord, tri_angles))

px = vertex_coord           #triangle vertex coordinates
trix = tri_angles           #triangles
nm = mm_normals             #marching cubes normals
snp = vert_norms            #surface _normals_np normals

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(px[:,0], px[:,1], px[:,2], triangles=trix, color='b')
#quiver: we want bottom = verticies, we want the top = normals 
ax.quiver(px[:,0],px[:,1],px[:,2], nm[:,0],nm[:,1],nm[:,2])
t = "Vertex normals \n Marching Cubes (b)"
plt.title(t)
plt.savefig('../task2/Trinorms_MarchingCubes.png') 


fig1 = plt.figure()
ax = fig1.gca(projection='3d')
ax.plot_trisurf(px[:,0], px[:,1], px[:,2], triangles=trix, color='r')
ax.quiver(px[:,0],px[:,1],px[:,2], snp[:,0],snp[:,1],snp[:,2],color='r')
t = "Vertex normals \n Surface_normals_np (r)"
plt.title(t)
plt.savefig('../task2/Trinorms_surface_norms_np.png') 
  
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(px[:,0], px[:,1], px[:,2], triangles=trix, color='b')
#quiver: we want bottom = verticies, we want the top = normals 
ax.quiver(px[:,0],px[:,1],px[:,2], nm[:,0],nm[:,1],nm[:,2])
ax.quiver(px[:,0],px[:,1],px[:,2], snp[:,0],snp[:,1],snp[:,2],color='r')
t = "Vertex normals \n Marching Cubes (b) \n Surface_normals_np (r)"
plt.title(t)
plt.savefig('../task2/Trinorms_MC_SNP.png') 


# %%
